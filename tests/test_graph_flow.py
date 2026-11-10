from __future__ import annotations

import asyncio

from asha_vani.voice.turn_runner import LangGraphTurnRunner, VerificationStatus


def test_negative_readback_keeps_visit_unverified_until_correction_and_yes(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    async def scenario() -> None:
        runner = LangGraphTurnRunner()
        first = await runner.run_turn(
            "flow-1", 1,
            "Pooja Devi, 32 weeks, BP 150 by 95, Hb 9.5, severe headache",
        )
        assert first.next_action == "CONFIRM_VALUES"
        assert first.verification is VerificationStatus.UNVERIFIED

        rejected = await runner.run_turn("flow-1", 2, "Nahi, BP galat hai")
        assert rejected.next_action == "CORRECT_VALUES"
        assert rejected.verification is VerificationStatus.UNVERIFIED

        corrected = await runner.run_turn("flow-1", 3, "BP 120 by 80 hai")
        assert corrected.visit_values.bp_systolic == 120
        assert corrected.visit_values.bp_diastolic == 80
        assert corrected.next_action == "CONFIRM_VALUES"
        assert corrected.verification is VerificationStatus.UNVERIFIED

        confirmed = await runner.run_turn("flow-1", 4, "Haan sahi hai")
        assert confirmed.is_confirmation_turn is True
        assert confirmed.verification is VerificationStatus.VERIFIED

    asyncio.run(scenario())


def test_critical_symptom_is_not_hidden_by_invalid_bp_repeat(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    async def scenario() -> None:
        runner = LangGraphTurnRunner()
        result = await runner.run_turn(
            "flow-2", 1, "BP 310 by 90 hai aur daura pada hai"
        )
        assert result.visit_values.bp_systolic is None
        assert result.triage_result.level.value == "CRITICAL"
        assert result.next_action == "REPEAT_VALUE"
        assert "PHC team" in result.reply_text

    asyncio.run(scenario())


def test_verified_visits_can_repeat_in_one_call_then_end(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    async def scenario() -> None:
        runner = LangGraphTurnRunner(require_full_checklist=True)
        first = await runner.run_turn(
            "multi-call", 1,
            "Pooja Devi Rampur pati Ramesh Kumar 32 weeks BP 120 by 80 Hb 11.2 koi takleef nahi",
        )
        assert first.next_action == "CONFIRM_VALUES"
        first_confirmed = await runner.run_turn("multi-call", 2, "Haan sahi hai")
        assert first_confirmed.verification is VerificationStatus.VERIFIED
        assert first_confirmed.next_action == "ASK_ANOTHER_PATIENT"

        next_visit = await runner.run_turn("multi-call", 3, "Haan")
        assert next_visit.visit_number == 2
        assert next_visit.next_action == "ASK_PATIENT_NAME"

        second = await runner.run_turn(
            "multi-call", 4,
            "Kavita Devi Belwa pati Sanjay Mahto 28 weeks BP 118 by 76 Hb 10.5 koi takleef nahi",
        )
        assert second.visit_number == 2
        assert second.next_action == "CONFIRM_VALUES"
        second_confirmed = await runner.run_turn("multi-call", 5, "Haan sahi hai")
        assert second_confirmed.verification is VerificationStatus.VERIFIED
        end = await runner.run_turn("multi-call", 6, "Nahi")
        assert end.next_action == "END_CALL"
        assert end.call_complete is True
        assert end.visit_number == 2

    asyncio.run(scenario())
