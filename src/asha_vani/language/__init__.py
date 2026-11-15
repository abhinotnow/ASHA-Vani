"""Multilingual locale loading, ASHA worker registry, and language routing."""

from asha_vani.language.locale import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    get_locale,
    lexicon,
    locale_meta,
    readback_field,
    symptom_label,
    t,
)

__all__ = [
    "DEFAULT_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    "get_locale",
    "lexicon",
    "locale_meta",
    "readback_field",
    "symptom_label",
    "t",
]

