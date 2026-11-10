"""Unit and integration tests for Person A's `src/asha_vani/voice/` modules."""

from __future__ import annotations

import asyncio
from array import array
import math

from asha_vani.schemas import SymptomCode, TriageLevel
from asha_vani.voice.audio import (
    FRAME_BYTES,
    SAMPLE_RATE_HZ,
    FrameBuffer,
    condition_telephone_audio,
    mix_snr_noise,
    mulaw_roundtrip,
    pcm16_to_mulaw,
    resample_pcm16,
    rms_energy,
    samples_to_bytes,
)
from asha_vani.voice.events import CallEventType, InMemoryEventSink
from asha_vani.voice.session import CallSession, HalfDuplexGate
from asha_vani.voice.stt import EndOfUtteranceDetector, GnaniPrismaSTT, STTChunk, strip_trailing_over
from asha_vani.voice.tts import FakeTTS, FixedScriptCache, FIXED_SCRIPTS, GnaniTimbreTTS
from asha_vani.voice.turn_runner import LangGraphTurnRunner, VerificationStatus


def _make_sine_pcm16(freq_hz: float = 1000.0, duration_s: float = 0.25, rate: int = SAMPLE_RATE_HZ) -> bytes:
    n = int(rate * duration_s)
    samples = array(
        "h",
        (int(12000 * math.sin(2.0 * math.pi * freq_hz * i / rate)) for i in range(n)),
    )
    return samples_to_bytes(samples)


def test_frame_buffer_outputs_exact_1024_byte_frames() -> None:
    fb = FrameBuffer()
    # Push 2500 bytes -> should yield 2 full 1024-byte frames and hold 452 bytes
    frames = fb.push(b"\x01\x00" * 1250)
    assert len(frames) == 2
    assert all(len(f) == FRAME_BYTES for f in frames)
    assert fb.buffered_bytes == 452

    # Flush pads the remainder with silence to exact 1024 bytes
    tail = fb.flush(pad_with_silence=True)
    assert tail is not None
    assert len(tail) == FRAME_BYTES
    assert fb.buffered_bytes == 0


def test_mulaw_roundtrip_and_snr_noise_calibration() -> None:
    clean = _make_sine_pcm16(freq_hz=800.0, duration_s=0.5)
    compressed = pcm16_to_mulaw(clean)
    # 8-bit mu-law is half the byte size of 16-bit PCM
    assert len(compressed) == len(clean) // 2

    roundtripped = mulaw_roundtrip(clean)
    assert len(roundtripped) == len(clean)
    # Signal energy is preserved within reasonable quantization tolerance
    assert abs(rms_energy(roundtripped) - rms_energy(clean)) / rms_energy(clean) < 0.08

    # 5 dB SNR adds more noise energy than 10 dB SNR
    noisy_10db = mix_snr_noise(clean, snr_db=10.0)
    noisy_5db = mix_snr_noise(clean, snr_db=5.0)
    assert rms_energy(noisy_5db) > rms_energy(noisy_10db) > rms_energy(clean)


def test_resample_and_telephone_conditioning() -> None:
    pcm_16k = _make_sine_pcm16(freq_hz=1000.0, duration_s=0.2, rate=16000)
    pcm_8k = resample_pcm16(pcm_16k, src_rate=16000, dst_rate=8000)
    assert len(pcm_8k) == len(pcm_16k) // 2

    conditioned = condition_telephone_audio(pcm_16k, src_rate=16000, snr_db=10.0)
    assert len(conditioned) == len(pcm_8k)


def test_eou_detector_flushes_on_over_keyword_and_strips_it() -> None:
    cleaned, had_over = strip_trailing_over("Pooja Devi ka BP 150 by 95 hai over.")
    assert had_over is True
    assert cleaned == "Pooja Devi ka BP 150 by 95 hai"

    eou = EndOfUtteranceDetector()
    # Partial without 'over' does not flush
    assert eou.feed(STTChunk(text="Pooja Devi ka BP", is_final=False)) is None
    # Partial ending with 'over' flushes immediately and strips 'over'
    out = eou.feed(STTChunk(text="Pooja Devi ka BP 150 by 95 hai OVER", is_final=False))
    assert out == "Pooja Devi ka BP 150 by 95 hai"


def test_half_duplex_gate_enforces_200ms_tail_mute() -> None:
    now = [100.0]
    gate = HalfDuplexGate(tail_mute_s=0.20, clock=lambda: now[0])
    assert gate.is_mic_muted is False

    gate.start_speaking()
    assert gate.is_mic_muted is True

    gate.stop_speaking()
    # 100 ms after TTS stops -> still muted inside 200 ms tail window
    now[0] = 100.10
    assert gate.is_mic_muted is True

    # 250 ms after TTS stops -> unmuted
    now[0] = 100.25
    assert gate.is_mic_muted is False


def test_fixed_script_cache_avoids_duplicate_synthesis() -> None:
    async def _run() -> None:
        fake_tts = FakeTTS(frames_per_reply=2)
        cached_tts = FixedScriptCache(fake_tts)
        greeting = FIXED_SCRIPTS["GREETING"]

        frames_1 = [f async for f in cached_tts.synthesize(greeting)]
        frames_2 = [f async for f in cached_tts.synthesize(greeting)]
        assert len(frames_1) == 2
        assert frames_1 == frames_2
        # Underlying TTS was only called once!
        assert len(fake_tts.spoken_texts) == 1

    asyncio.run(_run())


def test_call_session_end_to_end_turn_flow_and_edge_cases() -> None:
    async def _run() -> None:
        now = [10.0]
        sink = InMemoryEventSink()
        tts = FakeTTS(frames_per_reply=2)
        session = CallSession(
            "sess-101",
            caller="+919900000001",
            turn_runner=LangGraphTurnRunner(),
            tts=tts,
            sinks=[sink],
            tail_mute_s=0.20,
            clock=lambda: now[0],
        )

        await session.start()

        # Turn 1: Dialect Hindi numbers ("dedh sau by pachhanve" = 150/95) + severe headache + "over"
        t1 = await session.receive_stt_chunk(
            STTChunk(
                text=(
                    "Rampur Purani Basti se Pooja Devi patni Ramesh Kumar, "
                    "32 hafte, BP dedh sau by pachhanve hai aur tez sar dard hai over"
                ),
                is_final=False,
            ),
            stt_latency_ms=140.0,
        )
        assert t1 is not None
        assert t1.patient.full_name == "Pooja Devi"
        assert t1.patient.spouse_or_guardian_name == "Ramesh Kumar"
        assert t1.patient.village == "Rampur"
        assert t1.visit_values.bp_systolic == 150
        assert t1.visit_values.bp_diastolic == 95
        assert t1.visit_values.gestational_weeks == 32
        assert SymptomCode.SEVERE_HEADACHE in t1.visit_values.present_symptoms
        assert t1.triage_result.level is TriageLevel.CRITICAL
        assert "R2" in t1.triage_result.rules_fired
        assert t1.verification is VerificationStatus.UNVERIFIED

        # Immediately after TTS finishes (within 200ms tail mute), mic frames must be dropped!
        now[0] = 10.05
        assert session.push_mic_audio(b"\x01\x02" * 512) == []
        assert session.dropped_mic_bytes == 1024

        # Advance clock past 200ms tail mute -> mic accepts frames again
        now[0] = 10.50
        assert len(session.push_mic_audio(b"\x01\x02" * 512)) == 1

        # Turn 2: ASHA confirms readback -> flips from UNVERIFIED to VERIFIED
        t2 = await session.receive_stt_chunk("Haan ji bilkul sahi hai over", stt_latency_ms=95.0)
        assert t2 is not None
        assert t2.is_confirmation_turn is True
        assert t2.verification is VerificationStatus.VERIFIED
        assert t2.next_action == "ASK_ANOTHER_PATIENT"

        await session.end(reason="COMPLETED")

        # Verify EventSink received call_started, 2 turn_completed events, and call_ended
        assert [e.event_type for e in sink.events] == [
            CallEventType.CALL_STARTED,
            CallEventType.TURN_COMPLETED,
            CallEventType.TURN_COMPLETED,
            CallEventType.CALL_ENDED,
        ]
        assert sink.events[-1].payload["final_verification"] == "VERIFIED"

    asyncio.run(_run())


def test_ending_with_unconfirmed_visit_marks_it_incomplete() -> None:
    async def scenario() -> None:
        sink = InMemoryEventSink()
        session = CallSession("unfinished", turn_runner=LangGraphTurnRunner(), sinks=[sink])
        await session.start()
        out = await session.process_utterance("Pooja Devi Rampur BP 150 by 95")
        assert out is not None
        assert out.verification is VerificationStatus.UNVERIFIED
        ended = await session.end(reason="COMPLETED")
        assert ended.payload["final_verification"] == "INCOMPLETE"

    asyncio.run(scenario())


def test_call_session_safe_refusal_on_hard_invalid_bp_and_dropped_call() -> None:
    async def _run() -> None:
        sink = InMemoryEventSink()
        session = CallSession("sess-drop", sinks=[sink])
        await session.start()

        # Garbled BP 310/95 -> rejected by hard limit, routed to REPEAT_VALUE, never stored
        t1 = await session.process_utterance("Guddi Devi, 30 hafte, BP 310 by 95 hai over")
        assert t1 is not None
        assert t1.visit_values.bp_systolic is None
        assert t1.next_action == "REPEAT_VALUE"
        assert t1.verification is VerificationStatus.UNVERIFIED

        # Call drops before confirmation -> marked INCOMPLETE
        end_ev = await session.end(reason="DROPPED")
        assert end_ev.payload["final_verification"] == "INCOMPLETE"

    asyncio.run(_run())


def test_gnani_prisma_and_timbre_configs_and_devanagari_turn_flow() -> None:
    import pytest

    # Timbre v2.0 is retired by Gnani API and must raise ValueError immediately
    with pytest.raises(ValueError, match="timbre-v2.0"):
        GnaniTimbreTTS(model="timbre-v2.0")

    tts_client = GnaniTimbreTTS()
    assert tts_client.model == "timbre-v2.5"
    assert tts_client.voice == "Nalini"
    assert tts_client.language == "hi-IN"

    stt_client = GnaniPrismaSTT()
    assert stt_client.language_code == "hi-IN"
    assert stt_client.sample_rate == 8000
    assert stt_client.format == "transcribe"

    async def _run_devanagari() -> None:
        session = CallSession("sess-devanagari")
        await session.start()

        # Devanagari output from Gnani Prisma with ITN digits + Hindi 'ओवर'
        t1 = await session.receive_stt_chunk(
            "रामपुर पुरानी बस्ती से पूजा देवी पत्नी रमेश कुमार 32 हफ्ते बीपी 150 बाई 95 है और तेज सर दर्द है ओवर"
        )
        assert t1 is not None
        assert t1.patient.full_name == "Pooja Devi"
        assert t1.patient.spouse_or_guardian_name == "Ramesh Kumar"
        assert t1.patient.village == "Rampur"
        assert t1.patient.tola == "Purani Basti"
        assert t1.visit_values.gestational_weeks == 32
        assert t1.visit_values.bp_systolic == 150
        assert t1.visit_values.bp_diastolic == 95
        assert SymptomCode.SEVERE_HEADACHE in t1.visit_values.present_symptoms
        assert t1.triage_result.level is TriageLevel.CRITICAL
        assert "R2" in t1.triage_result.rules_fired
        assert t1.verification is VerificationStatus.UNVERIFIED

        # Devanagari confirmation
        t2 = await session.receive_stt_chunk("हाँ बिल्कुल सही है ओवर")
        assert t2 is not None
        assert t2.is_confirmation_turn is True
        assert t2.verification is VerificationStatus.VERIFIED

    asyncio.run(_run_devanagari())


def test_step_by_step_full_checklist_flow(monkeypatch) -> None:
    """Verify live interactive calls ask for Patient Name -> Spouse Name -> Weeks -> BP -> Hb -> Confirmation."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    async def _run_checklist() -> None:
        runner = LangGraphTurnRunner(require_full_checklist=True)
        session = CallSession("sess-checklist", turn_runner=runner)
        await session.start()

        # Turn 1: ASHA gives village + patient name ("Main Rampur mein Sunita Devi ke ghar hoon")
        t1 = await session.process_utterance("Main Rampur mein Sunita Devi ke ghar hoon")
        assert t1 is not None
        assert t1.patient.full_name == "Sunita Devi"
        assert t1.patient.village == "Rampur"
        assert t1.next_action == "ASK_SPOUSE_NAME"
        assert "pati (husband) ka naam" in t1.reply_text

        # Turn 2: ASHA gives bare husband name ("Rakesh Kumar") -> saved as spouse, preserves patient name
        t2 = await session.process_utterance("Rakesh Kumar")
        assert t2 is not None
        assert t2.patient.full_name == "Sunita Devi"
        assert t2.patient.spouse_or_guardian_name == "Rakesh Kumar"
        assert t2.next_action == "ASK_WEEKS"
        assert "kitne hafte (weeks)" in t2.reply_text

        # Turn 3: ASHA gives weeks ("28") -> asks for BP
        t3 = await session.process_utterance("28")
        assert t3 is not None
        assert t3.visit_values.gestational_weeks == 28
        assert t3.next_action == "ASK_BP"
        assert "BP (blood pressure)" in t3.reply_text

        # Turn 4: ASHA gives elevated BP ("BP 150 by 95") -> still asks for Hemoglobin & symptoms!
        t4 = await session.process_utterance("BP 150 by 95")
        assert t4 is not None
        assert t4.visit_values.bp_systolic == 150
        assert t4.visit_values.bp_diastolic == 95
        assert t4.next_action == "ASK_HB"
        assert "Hemoglobin (Hb)" in t4.reply_text

        # Turn 5: ASHA gives Hb + symptom -> full readback confirmation with all fields
        t5 = await session.process_utterance("Hb 9 point 5 hai aur tez sar dard hai")
        assert t5 is not None
        assert t5.visit_values.hb_g_dl == 9.5
        assert t5.next_action == "CONFIRM_VALUES"
        assert "marij ka naam Sunita Devi, pati ka naam Rakesh Kumar" in t5.reply_text
        assert "28 hafte" in t5.reply_text
        assert "BP 150 by 95" in t5.reply_text
        assert "Hemoglobin 9.5 g/dL" in t5.reply_text
        assert t5.verification is VerificationStatus.UNVERIFIED

        # Turn 6: ASHA confirms readback -> flips to VERIFIED
        t6 = await session.process_utterance("Haan sahi hai")
        assert t6 is not None
        assert t6.is_confirmation_turn is True
        assert t6.verification is VerificationStatus.VERIFIED
        assert t6.next_action == "ASK_ANOTHER_PATIENT"

    asyncio.run(_run_checklist())


def test_non_pregnant_and_dtmf_keypad_flow(monkeypatch) -> None:
    """Verify non-pregnant general checkups ('pregnant nahi hai') and DTMF keypad fallback ('005', '150*95', '1')."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    async def _run_non_pregnant() -> None:
        runner = LangGraphTurnRunner(require_full_checklist=True)
        session = CallSession("sess-np", turn_runner=runner)
        await session.start()

        # DTMF keypad '005' for ASHA ID
        t1 = await session.process_utterance("005")
        assert t1 is not None
        assert t1.asha_id == "ASHA-005"
        assert t1.next_action == "ASK_PATIENT_NAME"

        # Patient & Spouse
        await session.process_utterance("Belwa mein Kavita Devi ke ghar")
        t3 = await session.process_utterance("Sanjay Mahto")
        assert t3 is not None
        assert t3.next_action == "ASK_WEEKS"

        # Non-pregnant answer -> skips weeks and moves straight to ASK_BP!
        t4 = await session.process_utterance("Pregnant nahi hai, general checkup hai")
        assert t4 is not None
        assert t4.is_pregnant is False
        assert t4.visit_category == "GENERAL_VITALS"
        assert t4.next_action == "ASK_BP"

        # DTMF keypad '150*95' for BP 150/95 -> moves to ASK_HB
        t5 = await session.process_utterance("150*95")
        assert t5 is not None
        assert t5.visit_values.bp_systolic == 150
        assert t5.visit_values.bp_diastolic == 95
        assert t5.next_action == "ASK_HB"

        # Hb 6.5 -> moves to ASK_SYMPTOMS so stomach pain / danger signs are checked even for non-pregnant visits
        t6 = await session.process_utterance("6.5")
        assert t6 is not None
        assert t6.visit_values.hb_g_dl == 6.5
        assert t6.next_action == "ASK_SYMPTOMS"
        assert "pet mein dard" in t6.reply_text

        # Symptom check -> moves to CONFIRM_VALUES with Non-Pregnant Checkup
        t6b = await session.process_utterance("Koi takleef nahi")
        assert t6b is not None
        assert t6b.next_action == "CONFIRM_VALUES"
        assert "Non-Pregnant Checkup" in t6b.reply_text
        assert "diagnosis" not in t6b.reply_text.lower()

        # DTMF keypad '1' to confirm readback -> VERIFIED
        t7 = await session.process_utterance("1")
        assert t7 is not None
        assert t7.is_confirmation_turn is True
        assert t7.verification is VerificationStatus.VERIFIED

    asyncio.run(_run_non_pregnant())


def test_zero_weeks_and_spouse_surname_and_stomach_pain_edge_cases(monkeypatch) -> None:
    """Verify '0 weeks' / '0' for non-pregnant, spouse surname preservation ('Rakesh Yadav'), and 'pet dard' extraction."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    async def _run_edges() -> None:
        runner = LangGraphTurnRunner(require_full_checklist=True)
        tts = FakeTTS(frames_per_reply=2)
        session = CallSession("sess-edges", turn_runner=runner, tts=tts)
        await session.start()

        # Turn 1: Wife name ("Sunita Devi")
        t1 = await session.process_utterance("Rampur se Sunita Devi")
        assert t1 is not None
        assert t1.patient.full_name == "Sunita Devi"
        assert t1.next_action == "ASK_SPOUSE_NAME"

        # Turn 2: Husband name with shared surname ("Rakesh Yadav") must NOT overwrite wife's name!
        t2 = await session.process_utterance("Rakesh Yadav")
        assert t2 is not None
        assert t2.patient.full_name == "Sunita Devi"
        assert t2.patient.spouse_or_guardian_name == "Rakesh Yadav"
        assert t2.next_action == "ASK_WEEKS"

        # Turn 3: '0 weeks' must NOT trigger [4, 44] validation error; it sets is_pregnant=False and asks BP
        t3 = await session.process_utterance("0 weeks")
        assert t3 is not None
        assert t3.is_pregnant is False
        assert t3.visit_values.gestational_weeks is None
        assert not t3.validation_result.has_errors
        assert t3.next_action == "ASK_BP"

        # Turn 4: BP 120 by 80 -> asks Hb
        t4 = await session.process_utterance("BP 120 by 80")
        assert t4 is not None
        assert t4.next_action == "ASK_HB"

        # Turn 5: Hb 11.2 + stomach pain ("pet mein dard hai") -> EPIGASTRIC_PAIN triggers R2 URGENT
        t5 = await session.process_utterance("Hb 11.2 hai aur pet mein dard hai")
        assert t5 is not None
        assert SymptomCode.EPIGASTRIC_PAIN in t5.visit_values.present_symptoms
        assert "R2" in t5.triage_result.rules_fired
        assert t5.triage_result.level is TriageLevel.URGENT
        assert t5.next_action == "CONFIRM_VALUES"
        assert "marij ka naam Sunita Devi, pati ka naam Rakesh Yadav" in t5.reply_text
        assert "pet mein dard" in t5.reply_text
        assert "diagnosis" not in t5.reply_text.lower()

        # Turn 6: Inline correction at CONFIRM_VALUES ("a sir, pati ka naam galat hai. pati ka naam abhinav hai")
        t6 = await session.process_utterance("a sir, pati ka naam galat hai. pati ka naam abhinav hai")
        assert t6 is not None
        assert t6.patient.full_name == "Sunita Devi"
        assert t6.patient.spouse_or_guardian_name == "Abhinav"
        assert t6.next_action == "CONFIRM_VALUES"
        assert "Theek hai, badal diya: pati ka naam Abhinav" in t6.reply_text

        # Turn 7: Devanagari correction at CONFIRM_VALUES -> transliterated to English ("Abhinav Pranav Raj")
        t7 = await session.process_utterance("अरे सर पति का नाम गलत है पति का नाम अभिनव प्रणव राज है")
        assert t7 is not None
        assert t7.patient.full_name == "Sunita Devi"
        assert t7.patient.spouse_or_guardian_name == "Abhinav Pranav Raj"
        assert t7.next_action == "CONFIRM_VALUES"
        assert "Theek hai, badal diya: pati ka naam Abhinav Pranav Raj" in t7.reply_text

        # Verify full-call WAV builder produces a valid RIFF WAVE header + audio payload
        wav_bytes = session.build_call_wav()
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"
        assert len(wav_bytes) > 44

    asyncio.run(_run_edges())




