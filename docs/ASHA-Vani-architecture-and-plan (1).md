# ASHA-Vani: Architecture and Implementation Plan

Gnani AI Great Indian AI Internship Challenge. Deadline: **Nov 10, 2026, 11:59 PM IST**.
Status as of Oct 26: RAG Q&A on hold. LangGraph used for the call state machine.

---

## 1. The problem in one minute

India has about a million ASHA workers. After home visits they spend hours copying data into paper registers. Government apps assume a smartphone, good data and typing in a vernacular keyboard. Dangerous pregnancy signs (high BP with headache, bleeding, severe anemia) can reach a doctor days late.

**ASHA-Vani:** the ASHA phones a number after a visit and speaks naturally in Hindi/Hinglish. The system turns the call into a verified register record, runs a clinician-reviewed risk check, alerts the PHC, and reads key values back for confirmation. A human makes every final decision (referral, ambulance).

**What it is not:** an autonomous doctor, a replacement for RCH/ANMOL, or an automatic ambulance dispatcher. It is decision support and early warning.

---

## 2. Scope decisions

| Decision | Detail |
|---|---|
| In | Voice call to structured record, confirm-back, deterministic triage, safe failure, live PHC console, measured benchmark |
| On hold | RAG Q&A for ASHAs (kept as a future node, section 12) |
| Roadmap only | Citizen-facing line, hash-chained ledger, full role matrix, duplicate scans, Erlang staffing, escalation timers, real telephony |
| LLM | Groq Llama 3.3 70B behind a thin provider interface, Llama 3.1 8B as fallback. Evon has no public API, so it is not used |
| Turn logic | LangGraph |
| Data | Synthetic only. No real patient names, numbers or recorded calls |

---

## 3. Architecture

```
Browser phone simulator / replay script / (stretch) Twilio
   -> Ingest adapter: converts to 8kHz PCM16, 1024-byte frames
   -> CallSession (one per call, asyncio tasks + queues)
        -> Gnani Prisma streaming STT
        -> Turn queue (one utterance at a time per call)
        -> LangGraph turn graph
             -> LLM extraction -> validation -> patient matching
             -> provisional triage (rule engine)
             -> confirm-back -> commit
        -> Gnani Timbre TTS (fixed scripts for critical messages)
   -> Postgres: calls, visits, patients, cases, call_events, alerts (+ outbox)
   -> Event stream -> PHC control console (browser)
```

### Design principles

1. **The LLM extracts; code decides.** The model never assigns risk.
2. **Alert early, commit late.** URGENT/CRITICAL alerts go out marked UNVERIFIED. The clinical record becomes verified only after the ASHA confirms the readback.
3. **Refuse to guess.** Invalid or unclear input leads to a repeat prompt and nothing clinical is written.
4. **Identity never blocks an emergency.** Unresolved patients are saved as provisional and flagged for review.
5. **Humans own the final action.** The system alerts and recommends.
6. **Label everything simulated.** Ambulance dispatch, RCH sync and alerts are mocks and say so on screen.

---

## 4. Stack

| Layer | Choice | Notes |
|---|---|---|
| Backend | Python, FastAPI, asyncio WebSockets, Pydantic | |
| STT | Gnani Prisma realtime | 16-bit PCM, 8 or 16 kHz, 1024-byte frames |
| TTS | Gnani Timbre | Check whether v3.0 is on the API; docs recommend v2.5 |
| LLM | Groq Llama 3.3 70B; Llama 3.1 8B fallback | Called directly via SDK, not through LangChain models |
| Turn logic | LangGraph (StateGraph) | Only for per-utterance decision logic |
| Database | Supabase Postgres | RLS on, backend uses service key only |
| Frontend | React + Vite | Phone simulator and PHC console |
| Testing | pytest, jiwer, replay script | |

---

## 5. Key stages and implementation

### 5.1 Ingest
- Each source is an adapter that outputs 8kHz PCM16.
- Twilio sends base64 mu-law in 20 ms frames: decode to PCM16 and re-buffer into 1024-byte frames (64 ms) for Prisma. Python's `audioop` is removed in newer versions, so check your version.
- The browser simulator band-passes the mic to 300-3400 Hz and round-trips it through mu-law so Prisma hears telephone-grade audio. Include a noise toggle and a SIM tag on every call.

### 5.2 Speech to text
- One Prisma WebSocket per call.
- **Verify on the STT realtime docs before building:** header names (VAD, bias list), event types (partial/final), end-of-speech signalling, and whether any confidence value is returned. Do not design around an ASR confidence score unless it exists.
- Treat a finished utterance as one turn when the ASHA pauses.

### 5.3 CallSession and concurrency
Each call gets its own task group:

| Task | Does |
|---|---|
| Audio-in | Reads from the socket, converts, sends to Prisma |
| Transcript reader | Reads Prisma events, detects end of utterance, puts it on the turn queue |
| Turn worker | Takes one utterance at a time, runs the LangGraph graph, puts the reply on the speech queue |
| Speaker | Streams Timbre audio back; mutes audio-in during playback (half-duplex) |
| Outbox worker | Sends queued (simulated) alerts with retry |

Rules:
- Turns run strictly in order per call. Two extractions never race.
- Semaphores cap concurrent requests to Prisma, Groq and Timbre. A 429 falls back to the smaller model. If a cap is hit, play a polite "please call back".
- CPU-heavy audio work (resampling, any denoising) runs outside the event loop.
- Use an async DB driver, or run the sync Supabase client in a thread executor.
- Every write carries an idempotency key (session id + utterance sequence). A retry never creates a duplicate row or a second alert.
- Dropped call: save the visit as INCOMPLETE with the verified fields, play or send a callback prompt, and merge into the same visit if she calls back within a set window (default 6 hours).

### 5.4 Turn logic with LangGraph

**Pattern:** one graph invocation per utterance. The call's `thread_id` is the session id. State is saved between turns by a checkpointer (in-memory for development; Postgres-backed if you want resume after a dropped call). Each invocation reads the new utterance and the current `step`, runs one or more nodes, produces a reply, and ends the turn.

**What LangGraph handles:** per-utterance routing, state between turns, optional checkpoint/resume, and the graph diagram for the write-up.
**What stays outside it:** audio, WebSockets, Prisma/Timbre streaming, timeouts, cancellation, the outbox worker, and the rule engine (a plain function the nodes call).

**State (shape, not code):** session_id, asha, patient candidates and chosen patient, step, last utterance, extracted fields with per-field status, validation results, provisional triage (level, rules fired), verification status, reply to speak, retry counter.

**Nodes and routing:**

| Node | Job | Next |
|---|---|---|
| route_entry | Runs first on every utterance: deterministic red-flag keyword check, then routes by `step` | Emergency path if a red flag is heard; otherwise the node for the current step |
| greet | Identify ASHA from caller id, greet | identify_patient |
| identify_patient | Find the patient from the ASHA's roster | disambiguate if 2+ candidates; collect_vitals if one; create provisional patient if none |
| disambiguate | Ask which one ("Purani Basti wali ya Nayi?") | identify_patient on answer |
| collect_vitals | LLM extraction, then validation. Ask for any missing critical field | confirm when critical fields are present; repeat prompt on invalid input |
| provisional_triage | Run rule engine on extracted fields. If URGENT/CRITICAL, queue an alert marked UNVERIFIED | confirm |
| confirm | Read back name and values; interpret yes / correction / unclear | commit_verified on yes; collect_vitals on correction (re-extract, re-triage, send a correction notice if the level changed); callback flag if no answer |
| commit_verified | Mark visit COMMITTED_VERIFIED, speak the fixed referral or routine script | anything_else |
| anything_else | "Aur kisi ki jaanch?" | identify_patient for another patient; end_call otherwise |
| emergency | Fires the provisional alert and fixed script for a heard red flag, then continues collection | collect_vitals |
| handle_invalid | Plays the repeat prompt, increments the retry counter | Back to the current step; after repeated failures, save INCOMPLETE and flag callback |
| end_call | Close the call and write call_ended | END |

Rules for the graph:
- Critical and urgent messages are fixed scripts, never LLM-written.
- The graph never commits clinical fields before the confirm step.
- Nodes call the rule engine and the DB layer through small functions, so they can be unit-tested without LangGraph.
- Test the whole conversation as text first (no audio, no API calls), then attach speech.
- LangGraph's API changes fairly often. Check the current docs for the checkpointer and state-schema syntax before writing code.

### 5.5 Extraction and validation
- One LLM call per finished utterance returns the structured schema; Pydantic validates it.
- Number normalization covers dialect phrases (for example "dedh sau by pachhanve" means 150/95).
- Shared constants module:

| Value | Hard limit (reject, re-ask) | Soft limit (must confirm) |
|---|---|---|
| Systolic BP | below 60 or above 260 | outside 80-240 |
| Diastolic BP | below 30 or above 160 | outside 40-140 |
| Hemoglobin (g/dL) | below 2.0 or above 20.0 | outside 3.0-18.0 |
| Gestational weeks | below 1 or above 44 | outside 4-42 |

Systolic must exceed diastolic. A hard-limit failure is never stored.
- Do not preload the whole roster into the prompt. Pass only the matching candidates or the spoken text.

### 5.6 Patient matching
- Search only the calling ASHA's roster: name, then spouse or guardian name, then tola or house, then age.
- Handle transliteration (Devanagari from Prisma versus Latin in the roster, Pooja/Puja variants, honorifics like Devi).
- Two or more candidates: ask aloud. None: create a PROVISIONAL patient and ask for spouse name and tola.
- Never merge automatically.

### 5.7 Rule engine (draft, needs clinician review)

| Rule | Condition | Level |
|---|---|---|
| R1 | Systolic 160+ or diastolic 110+ | CRITICAL |
| R2 | BP 140/90+ with severe headache, visual disturbance, epigastric pain or convulsion | CRITICAL |
| R3 | BP 140/90+ without a severe feature (including swelling alone) | URGENT |
| R4 | Vaginal bleeding at 20+ weeks | CRITICAL |
| R5a | Hb below 7 | URGENT |
| R5b | Hb below 5 | CRITICAL |
| R6 | Convulsion or unconsciousness | CRITICAL |
| R7 | Labour pains or leaking fluid before 37 weeks | CRITICAL |
| R8 | Reduced fetal movement from 28 weeks | URGENT |

- Rules are independent; the highest level wins. Plain Python with a unit test per rule and per boundary value.
- A red flag with missing BP still triages. With no red flag and missing vitals the visit is ROUTINE and the system asks a follow-up.
- A possible red flag gets one yes/no probe; if the call ends first, raise URGENT for human review.
- Output wording is "screening result, not a diagnosis". 145/95 with headache is URGENT/CRITICAL referral, not "severe pre-eclampsia".
- Add a "clinically reviewed" badge and quote guideline section numbers only after a named clinician has checked them against the source.

### 5.8 Response (TTS)
- Fixed scripts for critical messages, referrals, repeat prompts and the greeting.
- Pre-synthesize static prompts once and replay them. This saves credits and latency.
- Readback includes patient name and spouse name, and reads numbers digit by digit.

### 5.9 Persistence and alerts
- Write the visit, case and call_events rows plus an outbox row in one transaction.
- The outbox worker sends the simulated alert (WhatsApp/SMS mock) with retry, and the readback never waits on it.
- Row-level security on every table; the backend uses the service-role key (server only), the console uses a login.

### 5.10 PHC control console (frontend)
- Live calls strip, streaming transcript, register card with per-field confidence and an UNVERIFIED marker, triage banner with the rule ID, case queue sorted CRITICAL then URGENT then ROUTINE, and actions (claim, notify Medical Officer, confirm dispatch as a human step, close).
- Claim is one atomic update that only succeeds while the case is NEW, so two people never work the same case.
- Partial transcripts travel over a temporary live channel and are not stored. Finished utterances and triage changes are stored, then pushed.
- After a disconnect, reload open cases and all events after the last sequence number.

### 5.11 Phone simulator (frontend)
- Dial/ring UI, mic capture with the 8kHz filter and noise toggle, TTS playback, preset scenario buttons that play recorded clips, and a SIM tag.
- Run it and the console side by side in two browser windows against the same backend.

---

## 6. Data model (seven tables)

| Table | Key fields |
|---|---|
| asha_workers | id, asha_code, phone, name, sub_centre, phc_id |
| patients | id, phc_id, asha_id, full_name, name_key, spouse_or_guardian_name, age, village, tola, house_no, rch_id (nullable, unique), status (ACTIVE, PROVISIONAL, NEEDS_REVIEW) |
| calls | id, session_id (unique), phc_id, asha_id, source (BROWSER_SIM, REPLAY, PSTN), started_at, ended_at, end_reason (COMPLETED, DROPPED, ABANDONED) |
| visits | id, call_id, patient_id, idempotency_key (unique), gestational_weeks, bp_systolic, bp_diastolic, hb, symptoms[], triage (ROUTINE, URGENT, CRITICAL), rules_fired[], verification (PENDING_CONFIRMATION, COMMITTED_VERIFIED, INCOMPLETE, REJECTED), field_confidence |
| cases | id, visit_id (unique), triage, status (NEW, CLAIMED, ACTION_TAKEN, CLOSED), claimed_by, claimed_at, closed_at |
| call_events | id, call_id, seq, event_type, payload, actor, created_at; insert-only |
| alerts | id, case_id, channel, recipient_role, status (QUEUED, SENT, FAILED), created_at, sent_at |

Events: call_started, partial_transcript (broadcast only), final_utterance, entities_extracted, triage_updated, readback_sent, confirmation_received, alert_sent, call_ended.

Use `spouse_or_guardian_name` everywhere and one triage enum (ROUTINE/URGENT/CRITICAL) everywhere.

---

## 7. Evaluation plan

- Write 50-100 scripted synthetic calls covering severe hypertension, anemia, bleeding, routine ANC, duplicate names, noisy audio and garbled audio.
- Generate audio with Timbre, and also record some of your own voices (TTS audio flatters STT).
- Degrade to 8kHz mu-law, then add noise at 10 dB and 5 dB SNR.
- Report per condition: WER, BP-extraction accuracy, red-flag recall (with the number of cases and a confidence bound), refusal-to-guess rate, and p50/p95 latency per stage (end of speech to final transcript to extraction to triage to first TTS audio).
- 100 red-flag cases with no misses only supports roughly 97% recall at 95% confidence. Do not claim 99% without about 300 cases.
- Load test your own server (10, 25, 50 simulated calls) using stubbed STT and LLM responses. Spend real API credits only on small runs; the 5,000 programme credits are of unknown unit.
- Leave every result blank until it is measured. Show only measured numbers in the video.

---

## 8. Work split and timeline

Day 1 (Oct 27): agree the event messages and the case-actions API so the frontend can start against mocks.

| Phase | Person A: voice and reliability | Person B: logic and data |
|---|---|---|
| Backend (Oct 27-Nov 1) | Ingest adapters, Prisma and Timbre clients, prompt cache, CallSession tasks and queues, concurrency limits, latency logging, replay and load scripts, evaluation harness | LangGraph state, nodes and routing (text-only first), extraction and validators, patient matching, rule engine with tests, DB schema, outbox and alert worker |
| Frontend (Nov 2-5) | Phone simulator, noise toggle, scenario buttons, SIM tag | PHC console: live calls, case card, queue, claim and actions |
| Together (Nov 5-10) | Benchmark results, video recording and post | Write-up, README, clinician review follow-up |

Time freed by the RAG pause goes to: more synthetic call scripts, the clinician review of the rule table, and hardening the confirm-back and safe-failure paths.

Check-points:
- Oct 29: thin loop works (speech in, extracted fields, rule result, spoken reply).
- Nov 1: confirm-back, safe failure and the first benchmark run work.
- Nov 5: record the video.
- Nov 6: post on LinkedIn and X with both hashtags and tag Gnani, so engagement has time to build.
- Nov 9: submit. Nov 10 is buffer.

Each of you is interviewed individually. Record your own contributions in the submission and be able to explain the other person's component at a high level.

---

## 9. Sixty-second video storyboard

| Time | On screen | Point |
|---|---|---|
| 0:00-0:10 | Simulated feature phone dials; the call appears on the console tagged SIM | Zero app, zero data entry |
| 0:10-0:25 | Hinglish with village noise; transcript streams and the register fills | Real-time extraction on telephone audio |
| 0:25-0:35 | Readback and confirmation; banner turns CRITICAL with the rule ID; alert goes out | Confirm-back and deterministic triage |
| 0:35-0:45 | Dispatcher claims the case and confirms the referral | A human owns the final action |
| 0:45-0:55 | Garbled clip; the system asks for a repeat | Safe failure |
| 0:55-1:00 | Card with measured results and clinical review status | Evidence, not claims |

---

## 10. Verify before building

1. Prisma realtime docs: headers, partial/final events, end-of-speech signal, confidence field (if any), concurrent-stream limit.
2. Timbre: whether v3.0 is on the API, and how to cancel a stream.
3. LangGraph: current StateGraph, checkpointer and routing syntax.
4. Full contest terms on the registration page, especially anything about reusing earlier work.
5. Groq rate limits and Supabase free-tier pausing (keep the project warm before the demo).
6. Free-tier hosting cold starts: record the video against the local backend.

## 11. Claims to avoid unless proven

- Data residency in India (the demo sends text to a cloud LLM; say which path it uses).
- "Audited by a medical board" or specific guideline section numbers (until a named clinician signs off).
- Latency, WER or accuracy figures not measured on your own run.
- "Real cellular call" (it is simulated telephony; the same pipeline accepts Twilio/Exotel media streams).
- Automatic ambulance dispatch (it is an alert and recommendation; a human confirms).

## 12. Cut list and parked items

If you fall behind, drop in this order: Twilio, sensitive-disclosure routing, escalation timers, duplicate scanning, second language, multi-role console. Keep the rule tests, confirm-back, safe failure and the benchmark.

**Parked: RAG Q&A.** When revived, add an `ask_question` node to the graph (route from `anything_else`) backed by a small curated corpus with retrieval, source-naming, abstention below a similarity threshold, and a separate test set. Danger-sign questions must route to the rule engine's fixed scripts, never to the LLM.
