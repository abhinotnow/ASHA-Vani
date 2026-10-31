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
from asha_vani.voice.stt import EndOfUtteranceDetector, STTChunk, strip_trailing_over
from asha_vani.voice.tts import FakeTTS, FixedScriptCache, FIXED_SCRIPTS
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
        assert t2.next_action == "COMMITTED_VERIFIED"

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

