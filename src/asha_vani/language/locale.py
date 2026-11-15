"""Locale loader and deterministic template helper for ASHA-Vani."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from asha_vani.schemas import SymptomCode

DEFAULT_LANGUAGE = "hi"
SUPPORTED_LANGUAGES: tuple[str, ...] = ("hi", "ta", "te")
_LOCALES_DIR = Path(__file__).resolve().parent / "locales"

_BCP47_TO_CODE: dict[str, str] = {
    "hi": "hi",
    "hi-in": "hi",
    "ta": "ta",
    "ta-in": "ta",
    "te": "te",
    "te-in": "te",
    "en-in": "hi",  # fallback for Romanized Hinglish when auto-detect labels en-IN
}


def normalize_language_code(code: str | None) -> str:
    """Normalize a short or BCP-47 language code to a supported locale code."""
    if not code:
        return DEFAULT_LANGUAGE
    cleaned = code.strip().lower()
    if cleaned in _BCP47_TO_CODE:
        return _BCP47_TO_CODE[cleaned]
    short = cleaned.split("-")[0]
    return short if short in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


@lru_cache(maxsize=16)
def get_locale(lang: str = DEFAULT_LANGUAGE) -> dict[str, Any]:
    """Load and cache a locale JSON dictionary; falls back to Hindi if unknown."""
    code = normalize_language_code(lang)
    path = _LOCALES_DIR / f"{code}.json"
    if not path.exists():
        path = _LOCALES_DIR / f"{DEFAULT_LANGUAGE}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def list_available_locales() -> list[str]:
    """Return sorted list of locale codes that have a JSON file on disk."""
    return sorted(p.stem for p in _LOCALES_DIR.glob("*.json"))


def locale_meta(lang: str = DEFAULT_LANGUAGE) -> dict[str, Any]:
    return dict(get_locale(lang).get("meta", {}))


def t(lang: str | None, key: str, **kwargs: Any) -> str:
    """Format a prompt string from `prompts.<key>` with Hindi fallback."""
    code = normalize_language_code(lang)
    prompts = get_locale(code).get("prompts", {})
    template = prompts.get(key)
    if template is None and code != DEFAULT_LANGUAGE:
        template = get_locale(DEFAULT_LANGUAGE).get("prompts", {}).get(key)
    if template is None:
        raise KeyError(f"Missing locale prompt key '{key}' for '{code}'")
    return template.format(**kwargs) if kwargs else template


def readback_field(lang: str | None, key: str, **kwargs: Any) -> str:
    """Format a readback fragment from `readback_fields.<key>` with Hindi fallback."""
    code = normalize_language_code(lang)
    fields = get_locale(code).get("readback_fields", {})
    template = fields.get(key)
    if template is None and code != DEFAULT_LANGUAGE:
        template = get_locale(DEFAULT_LANGUAGE).get("readback_fields", {}).get(key)
    if template is None:
        raise KeyError(f"Missing locale readback_fields key '{key}' for '{code}'")
    return template.format(**kwargs) if kwargs else template


def symptom_label(lang: str | None, code: SymptomCode | str) -> str | None:
    """Return the localized spoken label for a SymptomCode."""
    lang_code = normalize_language_code(lang)
    key = code.value if isinstance(code, SymptomCode) else str(code)
    labels = get_locale(lang_code).get("symptom_labels", {})
    if key in labels:
        return labels[key]
    return get_locale(DEFAULT_LANGUAGE).get("symptom_labels", {}).get(key)


def lexicon(lang: str | None = DEFAULT_LANGUAGE) -> dict[str, Any]:
    """Return the extraction/keyword lexicon dict for a language."""
    return dict(get_locale(normalize_language_code(lang)).get("lexicon", {}))

