# ASHA-Vani (आशा-वाणी)

**Voice-to-Register Decision Support & Early Warning System for Rural Maternal Healthcare**  
*Built for the Gnani AI Great Indian AI Internship Challenge*

---

## 1. The Problem

India has nearly **one million ASHA (Accredited Social Health Activist) workers** conducting doorstep maternal health visits across rural villages. After visits, they spend hours copying vitals into paper registers, while digital government apps assume smartphones, reliable mobile data, and vernacular typing. Dangerous pregnancy signs—such as **severe hypertension with headache (pre-eclampsia risk), severe anemia, or antepartum bleeding**—can reach a Primary Health Centre (PHC) doctor days late.

## 2. What ASHA-Vani Does

With **ASHA-Vani**, an ASHA worker dials a toll-free number from any basic feature phone after a home visit and speaks naturally in **Hindi / Hinglish**:

1. **Telephone Audio Ingest:** Normalizes call audio to **8 kHz, 16-bit mono PCM** in **1024-byte (64 ms) frames** (300–3400 Hz telephone bandpass + G.711 $\mu$-law conditioning).
2. **Streaming Speech-to-Text (Gnani Prisma):** Transcribes rural Hindi/Hinglish speech in real time, detecting turn completion via natural pause (~1.2s) or a spoken `"over"` trigger.
3. **Structured LLM Extraction (Groq Llama 3.3 70B / 8B fallback):** Extracts structured candidate fields (`gestational_weeks`, `bp_systolic`, `bp_diastolic`, `hb_g_dl`, `symptoms`, and patient details).
4. **Hard/Soft Clinical Validation:** Rejects implausible or corrupted numbers (`ERROR` values are cleared and routed to a repeat prompt—never stored).
5. **Deterministic Rule Engine (`R1–R8`):** Evaluates sanitized vitals and danger signs to assign `ROUTINE`, `URGENT`, or `CRITICAL` screening levels with explicit rule IDs. **The LLM never assigns clinical risk.**
6. **Digit-by-Digit Spoken Confirm-Back (Gnani Timbre TTS):** Reads captured values back to the ASHA worker under strict half-duplex speaker control. Records start as `UNVERIFIED` and flip to `VERIFIED` only after confirmation.
7. **Live PHC Register & Pipeline Inspector Console:** Updates the Primary Health Centre's register table after each turn, sorted by triage priority (`CRITICAL` → `URGENT` → `ROUTINE`), with a slide-over **Step-by-Step Pipeline Inspector** showing what Prisma heard, what the LLM extracted, validation checks, rule engine decisions, and stage latencies.

> **Safety Boundary:** ASHA-Vani is clinical decision support and early warning—**not** an autonomous doctor or diagnosis engine, and **not** an automatic ambulance dispatcher. A human at the PHC owns every final referral action. All demo telephony and alerts use **100% synthetic data** and are explicitly labeled **`SIM`**.

---

## 3. Repository Structure

```text
ASHA-Vani/
├── AGENTS.md                        # System map, module ownership & edge-case invariants
├── docs/
│   ├── ASHA-Vani-architecture-and-plan (1).md
│   ├── person-a-build.md            # Person A (Voice, Session, Nokia Simulator) spec & contracts
│   └── person-b-build.md            # Person B (Schemas, Validation, Triage, LangGraph) spec
├── src/asha_vani/
│   ├── __init__.py
│   ├── schemas.py                   # Pydantic models (ExtractedVisit, ValidationResult, TriageResult)
│   ├── validation.py                # Hard/soft clinical range & BP consistency checks
│   ├── triage.py                    # Deterministic maternal danger-sign rule engine (R1–R8)
│   ├── graph.py                     # LangGraph per-turn state machine (call_graph)
│   ├── server.py                    # FastAPI + WebSocket server (/simulator, /ws/call, /ws/console)
│   └── voice/                       # (Person A) Audio conditioning, STT/TTS, CallSession, TurnRunner, simulator.html
└── tests/
    ├── test_validation.py           # Boundary & consistency unit tests
    ├── test_triage.py               # Rule engine R1–R8 unit tests
    └── test_voice.py                # Voice pipeline, CallSession & TurnRunner unit tests
```

---

## 4. Quick Start

### Prerequisites
- Python **3.13+**
- [`uv`](https://docs.astral.sh/uv/) package manager

### 1. Install Dependencies & Run Tests
```bash
uv sync
uv run pytest
```

### 2. Run the Live Voice Server & 75/25 Nokia Simulator Workspace
```bash
uv run asha-vani
```
Then open **`http://127.0.0.1:8000/simulator`** (or open [`src/asha_vani/voice/simulator.html`](src/asha_vani/voice/simulator.html) directly in your browser):
- **Left 75%:** PHC Rampur Register table (priority sorting, village/triage filters, 1-click referral status pill, and the **Slide-Over Step-by-Step Turn Pipeline Inspector**).
- **Right 25%:** Classic **Nokia feature-phone simulator** with live laptop mic capture, village noise selector (`Clean / 10 dB / 5 dB SNR`), and 1-click synthetic call scenarios.

---

## 5. Team & Work Split

| Teammate | Focus Area | Key Modules |
|---|---|---|
| **Person A** | Voice transport, audio conditioning, Gnani Prisma/Timbre, `CallSession`, Nokia Simulator & Eval harness | `src/asha_vani/voice/`, `src/asha_vani/server.py`, `tests/test_voice.py` |
| **Person B** | Clinical schemas, validation, deterministic triage rules (`R1–R8`), LangGraph flow, LLM extraction & DB persistence | `src/asha_vani/schemas.py`, `validation.py`, `triage.py`, `graph.py` |

