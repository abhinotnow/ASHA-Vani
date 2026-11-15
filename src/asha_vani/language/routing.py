"""Per-call language resolution, script detection, and mid-call language switching."""

from __future__ import annotations

from enum import Enum
import re
from typing import Iterable

from pydantic import Field

from asha_vani.language.locale import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    lexicon,
    locale_meta,
    normalize_language_code,
)
from asha_vani.language.registry import AshaWorker, WorkerRegistry
from asha_vani.schemas import ProjectModel

BILINGUAL_AUTO_GREETING = (
    "Namaste! Vanakkam! Namaskaram! ASHA-Vani mein aapka swagat hai. "
    "Apni bhasha mein apna ASHA ID aur marij ka naam batayein."
)

DEFAULT_LANGUAGE_CANDIDATES: list[str] = ["hi-IN", "ta-IN", "te-IN"]

_ROMANIZED_HINTS: dict[str, tuple[str, ...]] = {
    "ta": ("vanakkam", "naan", "aamaa", "aama", "illai", "illa", "oor", "vaaram", "thalai vali", "vayiru"),
    "te": ("namaskaram", "nenu", "avunu", "kaadu", "kadu", "ooru", "vaaram", "tala noppi", "kadupu"),
    "hi": ("namaste", "didi", "mera", "marij", "gaon", "hafte", "haan", "nahi", "pati"),
}


class LanguageSource(str, Enum):
    REGISTERED = "REGISTERED"
    AUTO_DETECTED = "AUTO_DETECTED"
    SWITCHED = "SWITCHED"
    DEFAULT = "DEFAULT"


class CallLanguage(ProjectModel):
    """Language state attached to a single CallSession."""

    code: str = DEFAULT_LANGUAGE  # 'hi' | 'ta' | 'te' | 'auto'
    bcp47: str = "hi-IN"  # 'hi-IN' | 'ta-IN' | 'te-IN' | 'indic-auto'
    source: LanguageSource = LanguageSource.DEFAULT
    worker: AshaWorker | None = None
    shared_workers: list[AshaWorker] = Field(default_factory=list)
    locked: bool = True
    candidates: list[str] = Field(default_factory=lambda: list(DEFAULT_LANGUAGE_CANDIDATES))

    @property
    def effective_code(self) -> str:
        """Locale code to use for prompts while `code == 'auto'`."""
        return DEFAULT_LANGUAGE if self.code == "auto" else normalize_language_code(self.code)

    @classmethod
    def from_code(
        cls,
        code: str,
        *,
        source: LanguageSource = LanguageSource.DEFAULT,
        worker: AshaWorker | None = None,
        shared_workers: list[AshaWorker] | None = None,
        locked: bool = True,
    ) -> CallLanguage:
        norm = normalize_language_code(code)
        meta = locale_meta(norm)
        return cls(
            code=norm,
            bcp47=str(meta.get("bcp47", "hi-IN")),
            source=source,
            worker=worker,
            shared_workers=list(shared_workers or []),
            locked=locked,
        )


def resolve_call_language(caller: str | None, registry: WorkerRegistry) -> CallLanguage:
    """Resolve initial language from caller phone number, falling back to `indic-auto`."""
    matches = registry.lookup_phone(caller)
    if len(matches) == 1:
        worker = matches[0]
        return CallLanguage.from_code(
            worker.language,
            source=LanguageSource.REGISTERED,
            worker=worker,
            locked=True,
        )
    if len(matches) > 1:
        langs = {normalize_language_code(w.language) for w in matches}
        common = next(iter(langs)) if len(langs) == 1 else DEFAULT_LANGUAGE
        return CallLanguage.from_code(
            common,
            source=LanguageSource.REGISTERED if len(langs) == 1 else LanguageSource.DEFAULT,
            worker=None,
            shared_workers=matches,
            locked=True,
        )
    return CallLanguage(
        code="auto",
        bcp47="indic-auto",
        source=LanguageSource.DEFAULT,
        worker=None,
        locked=False,
        candidates=list(DEFAULT_LANGUAGE_CANDIDATES),
    )


def detect_script_language(text: str) -> str | None:
    """Return 'ta', 'te', or 'hi' by Unicode block majority, or None if Latin/empty."""
    if not text:
        return None
    counts = {
        "ta": sum(1 for ch in text if "\u0b80" <= ch <= "\u0bff"),
        "te": sum(1 for ch in text if "\u0c00" <= ch <= "\u0c7f"),
        "hi": sum(1 for ch in text if "\u0900" <= ch <= "\u097f"),
    }
    best_lang, best_count = max(counts.items(), key=lambda kv: kv[1])
    return best_lang if best_count > 0 else None


def detect_romanized_language(text: str) -> str | None:
    """Heuristic detection for Romanized Hinglish / Tanglish / Telglish."""
    low = text.lower()
    if not low.strip():
        return None
    scores: dict[str, int] = {}
    for code, hints in _ROMANIZED_HINTS.items():
        score = sum(1 for h in hints if re.search(rf"\b{re.escape(h)}\b", low))
        if score:
            scores[code] = score
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


def detect_utterance_language(
    text: str,
    *,
    detected_language_code: str | None = None,
) -> str:
    """Resolve language from Gnani `detected_language_code`, Unicode script, or Romanized hints."""
    if detected_language_code:
        cand = detected_language_code.strip().lower()
        short = cand.split("-")[0]
        if short in SUPPORTED_LANGUAGES:
            return short
    script = detect_script_language(text)
    if script is not None:
        return script
    roman = detect_romanized_language(text)
    if roman is not None:
        return roman
    return DEFAULT_LANGUAGE


def parse_switch_request(text: str, languages: Iterable[str] = SUPPORTED_LANGUAGES) -> str | None:
    """Match any locale's `switch_request` phrases (e.g. 'Tamil mein', 'Telugu lo', 'தமிழில்')."""
    low = text.lower()
    for code in languages:
        phrases = lexicon(code).get("switch_request", [])
        for phrase in phrases:
            if phrase.lower() in low:
                return code
    return None


def is_not_me_utterance(text: str, lang: str = DEFAULT_LANGUAGE) -> bool:
    """Detect if a caller says she is not the registered worker on that phone."""
    low = text.lower()
    phrases = list(lexicon(lang).get("is_not_me", []))
    if lang != DEFAULT_LANGUAGE:
        phrases.extend(lexicon(DEFAULT_LANGUAGE).get("is_not_me", []))
    return any(p.lower() in low for p in phrases)

