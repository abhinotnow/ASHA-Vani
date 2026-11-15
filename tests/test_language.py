"""Unit tests for locales, phone normalization, worker registry, routing, and transliteration."""

from __future__ import annotations

import json
from pathlib import Path
import re

from asha_vani.extraction import transliterate_to_english
from asha_vani.language.locale import get_locale, list_available_locales, t
from asha_vani.language.registry import (
    WorkerRegistry,
    mask_phone,
    normalize_asha_id,
    normalize_phone,
)
from asha_vani.language.routing import (
    LanguageSource,
    detect_script_language,
    detect_utterance_language,
    parse_switch_request,
    resolve_call_language,
)


def _placeholders(template: str) -> set[str]:
    return set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", template))


def test_locale_keys_and_placeholders_match_hindi() -> None:
    locales = list_available_locales()
    assert {"hi", "ta", "te"}.issubset(set(locales))

    base = get_locale("hi")
    base_prompts = set(base["prompts"].keys())
    base_readback = set(base["readback_fields"].keys())
    base_symptoms = set(base["symptom_labels"].keys())

    for code in ("ta", "te"):
        loc = get_locale(code)
        assert set(loc["prompts"].keys()) == base_prompts
        assert set(loc["readback_fields"].keys()) == base_readback
        assert set(loc["symptom_labels"].keys()) == base_symptoms
        for key in base_prompts:
            assert _placeholders(loc["prompts"][key]) == _placeholders(base["prompts"][key]), f"{code}.prompts.{key}"
        for key in base_readback:
            assert _placeholders(loc["readback_fields"][key]) == _placeholders(base["readback_fields"][key]), f"{code}.readback_fields.{key}"
        assert t(code, "ASK_BP")


def test_phone_normalization_and_masking() -> None:
    assert normalize_phone("09876543210") == "+919876543210"
    assert normalize_phone("919876543210") == "+919876543210"
    assert normalize_phone("+91 98765-43210") == "+919876543210"
    assert normalize_phone("9876543210") == "+919876543210"
    assert normalize_phone("04423456789") is None  # landline
    assert normalize_phone("UNKNOWN") is None

    assert mask_phone("+919876543210") == "+91 98•••••3210"
    assert mask_phone("UNKNOWN") == "UNKNOWN"
    assert normalize_asha_id("5") == "ASHA-005"
    assert normalize_asha_id("ASHA-101") == "ASHA-101"


def test_worker_registry_lookup_merge_offline_fallback_and_pending_links(tmp_path: Path) -> None:
    seed = tmp_path / "seed.json"
    local = tmp_path / "local.json"
    seed.write_text(
        json.dumps({
            "workers": [
                {
                    "asha_id": "ASHA-101",
                    "full_name": "Lakshmi Murugan",
                    "language": "ta",
                    "phones": [{"phone": "+919900000101", "label": "own"}],
                },
                {
                    "asha_id": "ASHA-005",
                    "full_name": "Geeta Devi",
                    "language": "hi",
                    "phones": [{"phone": "+919900000099", "label": "shared"}],
                },
                {
                    "asha_id": "ASHA-008",
                    "full_name": "Usha Devi",
                    "language": "hi",
                    "phones": [{"phone": "+919900000099", "label": "shared"}],
                },
            ]
        }),
        encoding="utf-8",
    )
    local.write_text(
        json.dumps({
            "workers": [
                {
                    "asha_id": "ASHA-101",
                    "full_name": "Lakshmi M (Local)",
                    "language": "ta",
                    "phones": [{"phone": "+919876543210", "label": "own"}],
                }
            ]
        }),
        encoding="utf-8",
    )

    class _FailingSupabase:
        def table(self, _name: str):
            raise RuntimeError("offline")

    reg = WorkerRegistry(seed_path=seed, local_path=local, supabase_client=_FailingSupabase())
    assert reg.lookup_phone("09999999999") == []
    one = reg.lookup_phone("09876543210")
    assert len(one) == 1 and one[0].asha_id == "ASHA-101" and one[0].full_name == "Lakshmi M (Local)"
    shared = reg.lookup_phone("+919900000099")
    assert {w.asha_id for w in shared} == {"ASHA-005", "ASHA-008"}

    # Pending phone link never auto-adds a phone row until approved
    link = reg.queue_phone_link("+919811112222", "ASHA-101", session_id="s-1")
    assert link is not None
    assert reg.lookup_phone("+919811112222") == []
    pending = reg.list_pending_links()
    assert len(pending) == 1
    assert "phone_e164" not in pending[0]
    assert pending[0]["phone_masked"] == "+91 98•••••2222"

    assert reg.approve_link(link["id"]) is True
    assert len(reg.lookup_phone("+919811112222")) == 1


def test_routing_script_detection_and_switch_requests() -> None:
    reg = WorkerRegistry()
    ta_lang = resolve_call_language("+919900000101", reg)
    assert ta_lang.code == "ta"
    assert ta_lang.source is LanguageSource.REGISTERED
    assert ta_lang.worker is not None and ta_lang.worker.asha_id == "ASHA-101"

    unk_lang = resolve_call_language("+919988776655", reg)
    assert unk_lang.code == "auto"
    assert unk_lang.bcp47 == "indic-auto"
    assert unk_lang.locked is False

    assert detect_script_language("வணக்கம் அக்கா") == "ta"
    assert detect_script_language("నమస్కారం అక్క") == "te"
    assert detect_script_language("नमस्ते दीदी") == "hi"
    assert detect_script_language("Vanakkam akka") is None

    assert detect_utterance_language("BP 150 by 95", detected_language_code="te-IN") == "te"
    assert detect_utterance_language("Vanakkam akka, thalai vali") == "ta"
    assert detect_utterance_language("Namaskaram akka, avunu") == "te"

    assert parse_switch_request("Tamil mein baat karo") == "ta"
    assert parse_switch_request("Telugu lo cheppandi") == "te"
    assert parse_switch_request("हिन्दी में") == "hi"


def test_tamil_and_telugu_transliteration_to_roman() -> None:
    ta_roman = transliterate_to_english("லட்சுமி மதுரை")
    te_roman = transliterate_to_english("పద్మ రెడ్డి గుంటూరు")
    assert ta_roman.isascii() and len(ta_roman.strip()) > 4
    assert te_roman.isascii() and len(te_roman.strip()) > 4

