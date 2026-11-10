# Person A: Voice Pipeline, Nokia Simulator, and Demo Workspace

See the [architecture plan](ASHA-Vani-architecture-and-plan%20(1).md) for full details. Person A owns speech transport, `CallSession` orchestration, and the simulated caller experience.

**Current status (Nov 10):** the live WebSocket path accepts browser PCM audio for Prisma and streams Timbre PCM audio back. Browser speech recognition/TTS remain fallbacks. Event names and payloads are a draft contract and still need Person A + Person B sign-off.

---

## 1. Call Flow

```text
Nokia Simulator / Replay script
   → Audio ingest (8 kHz PCM16, 300–3400 Hz bandpass, mu-law round-trip, 1024-byte / 64 ms frames)
   → Prisma streaming STT (final utterance on ~1.2s pause OR spoken "over" trigger)
   → TurnRunner handoff → Person B's call_graph (one completed utterance at a time)
   → Reply text → Timbre TTS (or cached fixed prompt) → Nokia speaker
   → EventSink (megaphone) → Turn-by-turn console updates / Latency logger / (Person B) Supabase
```

The graph owns conversation decisions. Audio streaming, buffering, playback, half-duplex muting, and call lifecycle stay in `CallSession`.

---

## 2. Design & UI Decisions Locked In (Oct 31)

### 2.1 Backend & Voice Pipeline (Person A)

| # | Area | Decision | Why / Notes for Person B |
|---|---|---|---|
| **A1** | **Folder layout** | All Person A modules live under `src/asha_vani/voice/` (`audio.py`, `events.py`, `turn_runner.py`, `stt.py`, `tts.py`, `session.py`) + `src/asha_vani/server.py` at root. | Zero overlap with Person B's root files (`schemas.py`, `validation.py`, `triage.py`, `graph.py`) so we never hit merge conflicts. |
| **A2** | **Graph handoff (`TurnRunner`)** | `CallSession` calls a thin `TurnRunner` protocol: `await run_turn(session_id, utterance_seq, transcript) -> TurnOutput`. The adapter invokes the graph with the transcript and call `thread_id`. | The graph now owns extraction and conversation state; the adapter maps graph state into the existing `TurnOutput` shape. |
| **A3** | **End-of-utterance detection** | Prisma final segments close turns using its VAD; the fallback browser-recognition path waits 5 seconds of silence or the spoken trigger **`"over"`**. | Prisma VAD tuning should be measured with demo speakers; the trigger remains useful in replay/manual mode. |
| **A4** | **Half-duplex speaker control** | Strict half-duplex on the backend: while TTS audio is streaming to the caller, `CallSession` drops/mutes incoming mic frames + holds a **200 ms tail buffer** after playback ends. | Prevents the mic from picking up TTS speaker echo and triggering false self-interruptions. |
| **A5** | **Audio conditioning & packages** | `audio.py` currently uses Python standard-library resampling/filtering, μ-law conversion, SNR mixing, and 1024-byte framing. Browser microphone audio is converted to 8 kHz mono PCM16 before streaming. | Keep the simulation conditioning distinct from any future real telephony codec adapter; no NumPy/SciPy dependency is currently required. |
| **A6** | **Turn-level events (`EventSink`)** | **No flickering live word-by-word partial transcripts on the dashboard.** `CallSession` emits `call_started`, `turn_completed`, and `call_ended` events to async sinks. | The Supabase sink is attached at the server boundary. Treat these names/payloads as provisional until both teammates approve them. |

### 2.2 UI & Demo Workspace Layout (Shared with Person B)

```text
+---------------------------------------------------------------+----------------------+
|  LEFT 75%: PHC Database Dashboard (Draggable Split)           |  RIGHT 25%:          |
|                                                               |  Nokia Feature Phone |
|  [Filter: Triage v] [Village v] [Status v] [Search Patient..] |  +----------------+  |
|  ------------------------------------------------------------ |  | SIM   00:24    |  |
|  Patient      Village  BP      Hb   Triage    Verify   Action |  | Calling PHC... |  |
|  *Pooja Devi  Rampur   150/95  9.2  CRITICAL  VERIFIED [Ref✓] |  +----------------+  |
|   Rinki K.    Rampur   118/76  11.0 ROUTINE   VERIFIED [Pend] |  | [Call]  [End]  |  |
|                                                               |  |  1    2    3   |  |
|  +----------------------------------------------------------+ |  +----------------+  |
|  | Slide-Over Drawer (when clicking "Pooja Devi"):          | |  Demo Controls:      |
|  | 1. Patient Vitals, Symptoms, Triage & Rules Fired (R2)   | |  [Mic] [Noise: 10dB] |
|  | 2. Step-by-Step Turn Pipeline Logs:                      | |  [Preset: Critical]  |
|  |    Turn 1:                                               | |  [Preset: Routine]   |
|  |     - STT Heard: "Pooja Devi ka BP 150 by 95..." (140ms) | |  [Preset: Garbled]   |
|  |     - LLM Extracted: {bp_systolic: 150, ...} (280ms)     | +----------------------+
|  |     - Validation: PASS (confirm high BP)                 |
|  |     - Rule Engine: CRITICAL — Rule R2 fired              |
|  |     - TTS Spoken: "Kripya confirm karein..."             |
|  +----------------------------------------------------------+ |
+---------------------------------------------------------------+----------------------+
```

| # | UI Decision | Detail |
|---|---|---|
| **U1** | **75% / 25% Draggable Split Screen** | **Left 75%:** PHC Database Dashboard. **Right 25%:** Classic **Nokia feature-phone simulator** (LCD screen showing call status, timer, `SIM` badge + physical Green Call / Red End buttons) with a compact **Demo Controls** box underneath (Mic toggle, Village Noise `Off / 10dB / 5dB`, and 1-click Preset Call Scenarios). |
| **U2** | **No live word-by-word transcript clutter** | The main dashboard is a clean structured database table. A row appears/updates **after each completed turn** marked `UNVERIFIED (In Call)`, and flips to `VERIFIED` once the ASHA confirms readback. |
| **U3** | **Sorting & Filtering** | Default sort by **Triage Priority (`CRITICAL` > `URGENT` > `ROUTINE`)**, plus quick filter pills/dropdowns for **Triage Level**, **Village/Sub-centre**, **Verification Status**, and a **Patient/ASHA Name search box**. |
| **U4** | **Click Patient → Slide-Over Inspector Drawer** | Clicking any patient's name slides open a detail panel inside the 75% area showing **(1) Full Vitals, Symptoms & Triage Rules Fired** and **(2) Step-by-Step Turn Pipeline Logs** (what Prisma STT heard → what the LLM extracted → Validation result → Rule Engine decision → Timbre TTS reply + latencies). |
| **U5** | **Minimal 1-Click Action Pill** | No multi-step dispatcher action boxes inside the drawer. Each row in the table has a simple 1-click status pill (`Pending` → `Referred ✓` / `Reviewed ✓` tagged `SIM`) so the human-in-the-loop step takes 1 click. |
| **U6** | **Visual Design System** | **Left 75% (PHC Clinic Dashboard):** `cal.md` + `vercel.md` Light Mode (`Geist` / `Inter` + `Geist Mono`, crisp white/stone `#f9fafb` surfaces, `1px` `#eaeaea` borders, high-contrast semantic pills for `CRITICAL` red / `URGENT` amber / `ROUTINE` green). **Right 25% (Village Nokia Dock):** Contrasting dark slate dock (`linear.app.md` `#111318`) so the Clinic Computer vs. Village Nokia Phone separation is immediately obvious on video. |

---

## 3. Handoff Contract with Person B

### 3.1 `TurnRunner` Input / Output (`CallSession` → `call_graph`)
When an utterance finishes, `CallSession` invokes `TurnRunner.run_turn(...)`:
- **Inputs:**
  - `session_id: str` (passed as `configurable={"thread_id": session_id}` to `call_graph`)
  - `utterance_seq: int` (`1, 2, 3...` strictly sequential per call; used for idempotency keys `f"{session_id}:{utterance_seq}"`)
  - `transcript: str` (final cleaned utterance text, with trailing `"over"` stripped)
- **Outputs (`TurnOutput`):**
  - `reply_text: str` (what Timbre TTS should speak)
  - `next_action: str` (`"REPEAT_VALUE"`, `"CONFIRM_VALUES"`, `"END_CALL"`, etc.)
  - `extracted_turn: ExtractedVisit` (what the LLM extracted on *this* turn, for the step-by-step inspector log)
  - `visit_values: ExtractedVisit` (accumulated sanitized values for the patient row)
  - `validation_result: ValidationResult`
  - `triage_result: TriageResult`

### 3.2 `CallEvent` Payloads (`CallSession` → `EventSink`s)
Every event carries `session_id`, `seq`, `event_type`, `timestamp`, `is_simulated=True`, and `payload`:
1. `call_started`: `{"caller": "+91...", "source": "BROWSER_SIM" | "REPLAY"}`
2. `turn_completed`:
   ```json
   {
     "utterance_seq": 1,
     "stt_transcript": "Pooja Devi ka BP dedh sau by pachhanve hai aur tez sar dard hai",
     "extracted_turn": {"bp_systolic": 150, "bp_diastolic": 95, "symptoms": [{"code": "SEVERE_HEADACHE", "presence": "PRESENT"}]},
     "visit_values": {"bp_systolic": 150, "bp_diastolic": 95, "symptoms": [{"code": "SEVERE_HEADACHE", "presence": "PRESENT"}]},
     "validation_issues": [],
     "triage": {"level": "CRITICAL", "rules_fired": ["R2"]},
     "next_action": "CONFIRM_VALUES",
     "verification": "UNVERIFIED",
     "reply_text": "Please confirm these details...",
     "latencies_ms": {"stt": 140.2, "graph": 310.5, "tts_first_byte": 95.0}
   }
   ```
3. `call_ended`: `{"reason": "COMPLETED" | "DROPPED" | "ABANDONED", "duration_s": 42.1}`

---

## 4. Remaining Checks (Person A + Shared)

- [ ] Run browser PCM → Prisma → graph → Timbre with real Gnani keys; tune VAD using representative demo speech.
- [ ] Approve the event and payload draft with Person B before treating persisted records or dashboard behavior as stable.
- [ ] Apply the initial Supabase migration and confirm console reload from saved calls and events.
- [ ] Measure per-stage latency and extraction/rule outcomes; keep thresholds labeled as draft until clinician review.
