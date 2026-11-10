from __future__ import annotations

import asyncio

import pytest

from asha_vani.extraction import extract_turn
from asha_vani.schemas import ConfirmationIntent, Presence, SymptomCode


def test_fallback_extracts_hinglish_clinical_values_and_patient(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = asyncio.run(extract_turn(
        "Rampur Purani Basti se Pooja Devi patni Ramesh Kumar, 32 hafte, "
        "BP dedh sau by pachhanave hai aur tez sar dard hai"
    ))

    assert result.patient.full_name == "Pooja Devi"
    assert result.patient.spouse_or_guardian_name == "Ramesh Kumar"
    assert result.patient.village == "Rampur"
    assert result.patient.tola == "Purani Basti"
    assert result.visit.gestational_weeks == 32
    assert (result.visit.bp_systolic, result.visit.bp_diastolic) == (150, 95)
    assert SymptomCode.SEVERE_HEADACHE in result.visit.present_symptoms


def test_fallback_extracts_devanagari_and_explicit_denial(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = asyncio.run(extract_turn("रामपुर से पूजा देवी, 32 हफ्ते, सिर दर्द नहीं है"))

    assert result.patient.full_name == "Pooja Devi"
    assert result.patient.village == "Rampur"
    report = next(s for s in result.visit.symptoms if s.code is SymptomCode.SEVERE_HEADACHE)
    assert report.presence is Presence.ABSENT


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [("Haan ji, sab sahi hai", ConfirmationIntent.YES), ("Nahi, BP galat hai", ConfirmationIntent.NO)],
)
def test_confirmation_intent_requires_confirmation_context(monkeypatch, transcript, expected) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = asyncio.run(extract_turn(transcript, expected_action="CONFIRM_VALUES"))
    assert result.confirmation_intent is expected


def test_bare_answer_is_interpreted_in_prompt_context(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = asyncio.run(extract_turn("28", expected_action="ASK_WEEKS"))
    assert result.visit.gestational_weeks == 28
