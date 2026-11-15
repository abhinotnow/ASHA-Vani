"""End-to-end multilingual call flows across Hindi, Tamil, and Telugu."""

from __future__ import annotations

import asyncio

import pytest

from asha_vani.language.registry import WorkerRegistry
from asha_vani.language.routing import LanguageSource, resolve_call_language
from asha_vani.schemas import TriageLevel
from asha_vani.voice.events import CallEventType, InMemoryEventSink
from asha_vani.voice.session import CallSession
from asha_vani.voice.stt import STTChunk
from asha_vani.voice.tts import FakeTTS
from asha_vani.voice.turn_runner import LangGraphTurnRunner, VerificationStatus


def test_registered_tamil_worker_full_flow_and_english_register() -> None:
    async def _run() -> None:
        reg = WorkerRegistry()
        sink = InMemoryEventSink()
        tts = FakeTTS()
        runner = LangGraphTurnRunner(require_full_checklist=False, registry=reg)
        session = CallSession(
            "ta-call-1",
            caller="+919900000101",
            turn_runner=runner,
            tts=tts,
            sinks=[sink],
            registry=reg,
        )
        assert session.language.code == "ta"
        assert "Lakshmi Murugan" in session.greeting_text()
        await session.start()

        # Tanglish + Tamil mixed utterance with CRITICAL BP + headache (R2)
        t1 = await session.process_utterance(
            "Patient name Kavitha Murugan, Madurai, husband Ramesh, 30 weeks, BP 160 by 110, Hb 10.5, romba thalai vali"
        )
        assert t1 is not None
        assert t1.language == "ta"
        assert t1.asha_id == "ASHA-101"
        assert t1.triage_result.level is TriageLevel.CRITICAL
        assert "R2" in t1.triage_result.rules_fired
        assert t1.patient.full_name == "Kavitha Murugan"
        assert t1.patient.village == "Madurai"
        assert "confirm" in t1.reply_text.lower()

        # Confirm in Tamil ("aamaa" / "ஆமா")
        t2 = await session.process_utterance("aamaa")
        assert t2 is not None
        assert t2.verification is VerificationStatus.VERIFIED
        assert "நன்றி அக்கா" in t2.reply_text

    asyncio.run(_run())


def test_registered_telugu_worker_full_flow() -> None:
    async def _run() -> None:
        reg = WorkerRegistry()
        sink = InMemoryEventSink()
        tts = FakeTTS()
        runner = LangGraphTurnRunner(require_full_checklist=False, registry=reg)
        session = CallSession(
            "te-call-1",
            caller="+919900000201",
            turn_runner=runner,
            tts=tts,
            sinks=[sink],
            registry=reg,
        )
        assert session.language.code == "te"
        assert "Padma Reddy" in session.greeting_text()
        await session.start()

        t1 = await session.process_utterance(
            "Patient name Sita Devi, Guntur, husband Mohan, 30 weeks, BP 160 by 110, Hb 10.5, tala noppi"
        )
        assert t1 is not None
        assert t1.language == "te"
        assert t1.triage_result.level is TriageLevel.CRITICAL
        assert "R2" in t1.triage_result.rules_fired
        assert "confirm" in t1.reply_text.lower()

        t2 = await session.process_utterance("avunu")
        assert t2 is not None
        assert t2.verification is VerificationStatus.VERIFIED
        assert "ధన్యవాదాలు అక్క" in t2.reply_text

    asyncio.run(_run())


def test_unknown_caller_auto_detects_tamil_and_queues_pending_link() -> None:
    async def _run() -> None:
        reg = WorkerRegistry()
        sink = InMemoryEventSink()
        tts = FakeTTS()
        runner = LangGraphTurnRunner(require_full_checklist=True, registry=reg)
        session = CallSession(
            "auto-call-1",
            caller="+919988776655",
            turn_runner=runner,
            tts=tts,
            sinks=[sink],
            registry=reg,
        )
        assert session.language.code == "auto"
        assert not session.language.locked
        await session.start()

        # First STT chunk arrives with Tamil script & detected_language_code="ta-IN"
        t1 = await session.receive_stt_chunk(
            STTChunk(text="வணக்கம் ASHA ID 101", is_final=True, detected_language_code="ta-IN")
        )
        assert t1 is not None
        assert session.language.code == "ta"
        assert session.language.locked is True
        assert sink.by_type(CallEventType.LANGUAGE_CHANGED)
        # Pending phone link queued, not auto-added to worker phones
        assert reg.lookup_phone("+919988776655") == []
        assert any(p["claimed_asha_id"] == "ASHA-101" for p in reg.list_pending_links())

    asyncio.run(_run())


def test_mid_call_keypad_9_switch_preserves_state() -> None:
    async def _run() -> None:
        reg = WorkerRegistry()
        sink = InMemoryEventSink()
        tts = FakeTTS()
        runner = LangGraphTurnRunner(require_full_checklist=True, registry=reg)
        session = CallSession(
            "switch-call-1",
            caller="+919900000001",
            turn_runner=runner,
            tts=tts,
            sinks=[sink],
            registry=reg,
        )
        await session.start()

        t1 = await session.process_utterance("marij ka naam Sunita Devi, gaon Rampur")
        assert t1 is not None
        assert t1.patient.full_name == "Sunita Devi"
        assert t1.next_action == "ASK_SPOUSE_NAME"

        # Press 9 -> language menu, then 3 -> Telugu
        await session.process_utterance("9")
        assert "Telugu" in tts.spoken_texts[-1]
        await session.process_utterance("3")
        assert session.language.code == "te"
        assert session.language.source is LanguageSource.SWITCHED
        assert "భర్త" in tts.spoken_texts[-1]  # repeats ASK_SPOUSE_NAME in Telugu

        # Next turn continues in Telugu with Sunita Devi preserved
        t2 = await session.process_utterance("Ramesh")
        assert t2 is not None
        assert t2.language == "te"
        assert t2.patient.full_name == "Sunita Devi"
        assert t2.patient.spouse_or_guardian_name == "Ramesh"

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("caller", "utterance"),
    [
        ("+919900000001", "marij Sunita Devi, Rampur, pati Mohan, 30 weeks, BP 160 by 110, Hb 10.5, tez sar dard"),
        ("+919900000101", "patient Kavitha Devi, Madurai, husband Mohan, 30 weeks, BP 160 by 110, Hb 10.5, thalai vali"),
        ("+919900000201", "patient Sita Devi, Guntur, husband Mohan, 30 weeks, BP 160 by 110, Hb 10.5, tala noppi"),
    ],
)
def test_triage_is_language_invariant_for_identical_clinical_facts(caller: str, utterance: str) -> None:
    async def _run() -> None:
        reg = WorkerRegistry()
        runner = LangGraphTurnRunner(require_full_checklist=False, registry=reg)
        lang = resolve_call_language(caller, reg)
        out = await runner.run_turn(f"s-{lang.code}", 1, utterance, language=lang.code)
        assert out.triage_result.level is TriageLevel.CRITICAL
        assert "R2" in out.triage_result.rules_fired

    asyncio.run(_run())
