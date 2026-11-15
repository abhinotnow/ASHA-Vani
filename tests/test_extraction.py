from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from asha_vani.extraction import extract_turn, redact_pii
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


def test_redact_pii_masks_phone_and_aadhaar_but_preserves_clinical_numbers() -> None:
    text = (
        "ASHA 005 phone +91 9876543210 aadhaar 1234 5678 9012, "
        "32 hafte, BP 150/95, Hb 8.5"
    )
    redacted = redact_pii(text)
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_AADHAAR]" in redacted
    assert "9876543210" not in redacted
    assert "1234 5678 9012" not in redacted
    assert "ASHA 005" in redacted
    assert "32 hafte" in redacted
    assert "BP 150/95" in redacted
    assert "Hb 8.5" in redacted


def test_evon_primary_extraction_and_thinking_disabled(monkeypatch) -> None:
    monkeypatch.setenv("EVON_BASE_URL", "https://evon.example.com/v1")
    monkeypatch.setenv("EVON_MODEL", "gnani/gnani-evon-v3.3-30B-A3B")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    captured: dict[str, object] = {}
    mock_content = (
        "<think>checking internal steps</think>\n"
        "```json\n"
        + json.dumps({
            "patient": {
                "full_name": "Sunita Devi",
                "village": "Rampur",
                "extra_ignored_field": "drop_me",
            },
            "visit": {
                "gestational_weeks": 30,
                "bp_systolic": 160,
                "bp_diastolic": 100,
                "hb_g_dl": 6.8,
                "symptoms": [
                    {"code": "SEVERE_HEADACHE", "presence": "PRESENT", "phrase": "tez sar dard", "extra": 1}
                ],
                "unexpected_visit_key": 123,
            },
            "confirmation_intent": "UNCLEAR",
            "triage_guess_ignored": "CRITICAL",
        })
        + "\n```"
    )

    async def fake_post(self, url: str, *, json: dict | None = None, headers: dict | None = None):
        captured["url"] = url
        captured["payload"] = json
        captured["headers"] = headers
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": mock_content}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(
        extract_turn("Sunita Devi phone +91 9876543210, 30 hafte, BP 160/100, Hb 6.8, tez sar dard")
    )

    assert captured["url"] == "https://evon.example.com/v1/chat/completions"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    user_msg = payload["messages"][1]["content"]
    assert "9876543210" not in user_msg
    assert "[REDACTED_PHONE]" in user_msg
    assert result.patient.full_name == "Sunita Devi"
    assert result.patient.village == "Rampur"
    assert result.visit.gestational_weeks == 30
    assert (result.visit.bp_systolic, result.visit.bp_diastolic) == (160, 100)
    assert result.visit.hb_g_dl == 6.8
    assert SymptomCode.SEVERE_HEADACHE in result.visit.present_symptoms


def test_evon_timeout_falls_back_to_local_extractor(monkeypatch) -> None:
    monkeypatch.setenv("EVON_BASE_URL", "https://evon.example.com/v1")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    async def timeout_post(self, url: str, *, json: dict | None = None, headers: dict | None = None):
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(httpx.AsyncClient, "post", timeout_post)

    result = asyncio.run(
        extract_turn("Rampur se Pooja Devi, 32 hafte, BP dedh sau by pachhanave hai aur tez sar dard hai")
    )
    assert result.patient.full_name == "Pooja Devi"
    assert result.patient.village == "Rampur"
    assert result.visit.gestational_weeks == 32
    assert (result.visit.bp_systolic, result.visit.bp_diastolic) == (150, 95)
    assert SymptomCode.SEVERE_HEADACHE in result.visit.present_symptoms
