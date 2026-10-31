# Person A: Voice Pipeline, Nokia Simulator, and Demo Workspace

See the [architecture plan](ASHA-Vani-architecture-and-plan%20(1).md) for full details. Person A owns speech transport, `CallSession` orchestration, and the simulated caller experience.

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
| **A2** | **Graph handoff (`TurnRunner`)** | `CallSession` calls a thin `TurnRunner` protocol: `await run_turn(session_id, utterance_seq, transcript) -> TurnOutput` with an adapter wrapping `call_graph`. | Lets Person A test the full voice/session loop immediately while Person B finishes wiring the LLM extraction node in `graph.py`. Once `graph.py` accepts `utterance: str` directly, only the adapter changes. |
| **A3** | **End-of-utterance detection** | **Natural pause** (~1.2s silence / Prisma end-of-speech event) **OR** spoken trigger word **`"over"`** at the end of a phrase (flushes turn immediately and strips `"over"` from the transcript). | Gives reliable turn boundaries in both quiet and noisy demos without cutting the ASHA off mid-sentence. |
| **A4** | **Half-duplex speaker control** | Strict half-duplex on the backend: while TTS audio is streaming to the caller, `CallSession` drops/mutes incoming mic frames + holds a **200 ms tail buffer** after playback ends. | Prevents the mic from picking up TTS speaker echo and triggering false self-interruptions. |
| **A5** | **Audio conditioning & packages** | Use `numpy` + `scipy` + pure-Python/numpy G.711 mu-law codec (since Python 3.13 removed `audioop`). Both browser simulator and backend enforce **8 kHz, 16-bit mono PCM, 300–3400 Hz bandpass, mu-law round-trip, optional SNR noise (10 dB / 5 dB), and 1024-byte (64 ms) frames**. | Guarantees that live browser calls and offline WAV replay/benchmark runs go through the exact same telephone-grade pipeline. |
| **A6** | **Turn-level events (`EventSink`)** | **No flickering live word-by-word partial transcripts on the dashboard.** `CallSession` emits structured turn-level events (`call_started`, `turn_completed`, `call_ended`) to registered async `EventSink` callbacks. | `CallSession` never imports Supabase directly. Right now sinks push to the live WebSocket and latency logger; Person B can attach a Supabase/outbox sink to the same hook with zero changes to `CallSession`. |

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
     "verification": "PENDING_CONFIRMATION",
     "reply_text": "Please confirm these details...",
     "latencies_ms": {"stt": 140.2, "graph": 310.5, "tts_first_byte": 95.0}
   }
   ```
3. `call_ended`: `{"reason": "COMPLETED" | "DROPPED" | "ABANDONED", "duration_s": 42.1}`

---

## 4. Step-by-Step Build Checklist (Person A)

- [ ] **Step 1: Audio pipeline (`src/asha_vani/voice/audio.py` + `tests/test_audio.py`)**
  - Add `numpy`, `scipy`, `jiwer` to `pyproject.toml`.
  - Resampler to 8 kHz mono PCM16, 300–3400 Hz Butterworth bandpass filter, G.711 mu-law encode/decode round-trip, SNR noise mixer (10 dB / 5 dB), and 1024-byte (64 ms) frame chunker.
- [ ] **Step 2: Events & TurnRunner contract (`src/asha_vani/voice/events.py`, `src/asha_vani/voice/turn_runner.py`)**
  - Typed `CallEvent`, `EventSink` protocol, in-memory/WebSocket broadcaster, and `LangGraphTurnRunner` adapter wrapping Person B's `call_graph`.
- [ ] **Step 3: STT & TTS interfaces + Fakes + End-of-Utterance detector (`src/asha_vani/voice/stt.py`, `src/asha_vani/voice/tts.py`)**
  - Streaming STT/TTS protocols, pause + `"over"` end-of-utterance detector, fixed prompt audio cache, and fake STT/TTS for deterministic tests & load testing.
- [ ] **Step 4: `CallSession` orchestrator (`src/asha_vani/voice/session.py` + `tests/test_session.py`)**
  - Ordered turn queue, half-duplex speaker gate (+ 200 ms tail mute), concurrency semaphores, dropped-call handling, and per-stage latency logging.
- [ ] **Step 5: Gnani Prisma STT & Timbre TTS live clients**
  - Implement against verified Gnani API docs (headers, events, audio framing).
- [ ] **Step 6: FastAPI server + 75/25 Split Demo Workspace (`src/asha_vani/server.py`, `tools/replay.py`)**
  - `/ws/call`, `/ws/console`, `/simulator` with the 75% Database + Slide-Over Turn Inspector and 25% Nokia Feature-Phone Simulator + Demo Controls.
