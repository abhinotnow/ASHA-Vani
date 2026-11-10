# ASHA-Vani: 1-Page Solution Pitch (Gnani.ai Submission)

> **One-Line Summary:** ASHA-Vani turns any rural 2G feature phone or PSTN line into a voice-verified clinical register and maternal early-warning system powered by **Gnani Prisma STT**, **Gnani Timbre v2.5 TTS**, a stateful **LangGraph** intake pipeline, and a **deterministic clinical rule engine (`R1–R8`)**.

---

## 1. The Problem: Paper Registers & Delayed Rural Escalation
India's **1M+ ASHA workers** conduct doorstep maternal and general health visits across rural villages, often operating on basic feature phones in high-noise environments. Vital readings (blood pressure, hemoglobin, gestational weeks, danger signs) are written into paper registers and batch-entered days later at the Primary Health Centre (**PHC**). By the time severe preeclampsia (`BP >= 160/110 mmHg`) or critical anemia (`Hb < 5.0 g/dL`) is reviewed by a Medical Officer, the window for early referral has often closed.

## 2. The Solution: ASHA-Vani
An ASHA worker dials into **ASHA-Vani** (from a physical phone via Exotel PSTN or the built-in 8 kHz Nokia feature-phone simulator) and speaks naturally in **Hindi or Hinglish** (with DTMF keypad fallback `0–9, *, #` for extreme noise). Across a guided 7-step conversation, ASHA-Vani:
1. Streams 8 kHz telephone audio through **Gnani Prisma STT** to transcribe code-mixed Hindi/Hinglish and Devanagari numerals.
2. Extracts structured clinical fields into a stateful **LangGraph** conversation (`CallState`) using Groq Llama 3.3 70B (with 8B/regex fallback).
3. Enforces **hard physiological bounds** (`validation.py`) and runs a **deterministic clinical rule engine (`R1–R8`)** (`triage.py`).
4. Speaks a **digit-by-digit Hindi/Hinglish confirm-back** via **Gnani Timbre v2.5 TTS** and commits the record as `VERIFIED` only after the ASHA confirms.
5. Broadcasts turn-by-turn updates to the **PHC Clinical Register Console** for 1-click Medical Officer home dispatch (`108` Ambulance, ANM Mobile, or Home Follow-up).

## 3. Core Gnani.ai Voice Integration (`src/asha_vani/voice/`)
- **8 kHz Telephony Conditioning (`voice/audio.py`):** Resamples incoming audio to 8 kHz mono PCM16, applies 300–3400 Hz telephone bandpass filtering, $\mu$-law companding, exact **1024-byte (64 ms)** framing, and calibrated SNR noise injection (`15 dB` and `5 dB`) to test rural acoustic resilience.
- **Gnani Prisma Streaming STT (`voice/stt.py`):** Real-time streaming recognition with dual End-of-Utterance (**EOU**) detection (`~1.2s` silence window or trailing `"over"` / `"ओवर"` keyword, stripped before extraction).
- **Gnani Timbre v2.5 TTS (`voice/tts.py`):** Low-latency 8 kHz PCM/WAV synthesis (`Poorvi` / `Nalini` voices) with an in-memory fixed-script cache so recurring greeting and follow-up prompts incur zero redundant latency or API cost.
- **Half-Duplex Anti-Echo Gate (`voice/session.py`):** Drops all incoming microphone frames while Timbre TTS plays plus a strict **200 ms acoustic tail mute** after playback finishes, preventing loudspeaker echo loops.

## 4. Clinical Safety Invariants ("LLM Extracts; Deterministic Code Decides")
1. **Zero LLM Diagnosis:** The LLM only extracts structured candidate slots (`ExtractedVisit`); risk classification is 100% deterministic (`triage.py`).
2. **Refuse to Guess:** Impossible ASR artifacts (e.g., `BP 420/110 mmHg`, `Hb 28 g/dL`, or `systolic <= diastolic`) are rejected as `ERROR`, cleared to `None`, and routed to a targeted spoken repeat prompt without masking co-occurring critical symptoms.
3. **Alert Early, Commit Late:** High-risk cases appear on the PHC console on Turn 1 marked `UNVERIFIED (In Call)`, flipping to `VERIFIED` only after the ASHA confirms the spoken readback (*"Haan, sahi hai"*).

## 5. Deterministic Triage Rule Engine (`R1–R8`)

| Rule | Risk Level | Deterministic Trigger Condition |
|---|---|---|
| `R1` | **CRITICAL** | Severe hypertension: `BP >= 160` systolic or `>= 110` diastolic |
| `R2` | **CRITICAL** | Elevated BP (`>= 140/90`) + severe headache, blurred vision, epigastric pain, or facial edema |
| `R3` | **URGENT** | Moderate hypertension (`>= 140/90`) without severe features |
| `R4` | **CRITICAL** | Antepartum vaginal bleeding at `>= 20 weeks` |
| `R5` | **CRITICAL / URGENT** | Severe anemia: `Hb < 5.0 g/dL` (**CRITICAL**); `5.0 <= Hb < 7.0 g/dL` (**URGENT**) |
| `R6` | **CRITICAL** | Eclampsia convulsion/seizure (`CONVULSION`) or `UNCONSCIOUS` |
| `R7` | **CRITICAL** | Preterm labour pains or leaking fluid at `< 37 weeks` |
| `R8` | **URGENT** | Reduced fetal movement at `>= 28 weeks` |

## 6. Verification & Live Routes (`69 passed` in Pytest)
- **Automated Test Suite:** `69 passed` across validation boundaries, `R1–R8` triage rules, Devanagari/Hinglish extraction, 1024-byte framing, 200 ms half-duplex gating, and Supabase persistence.
- **Live Routes:** `/furrow` (Hallmark Furrow Field Register & Handset UI), `/simulator` (Classic Clinical Register UI), `/pitch` (1-Page Print-Ready A4 Solution Pitch), `/ws/call`, and `/ws/console`.

