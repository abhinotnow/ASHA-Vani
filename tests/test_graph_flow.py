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
