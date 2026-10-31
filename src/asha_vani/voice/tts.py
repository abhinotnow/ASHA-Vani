"""Text-to-Speech (TTS) interface, Fixed Safety Script Cache, and Fakes.

Architecture Section 5.8:
- Fixed scripts for critical messages, referrals, repeat prompts, and greeting.
- Pre-synthesize static prompts once and cache them to save API credits and latency.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from asha_vani.voice.audio import FRAME_BYTES

FIXED_SCRIPTS: dict[str, str] = {
    "GREETING": (
        "Namaste didi, ASHA-Vani mein aapka swagat hai. "
        "Kripya marij ka naam, hafte, BP aur lakshan batayein."
    ),
    "REPEAT_NOISE": (
        "Mujhe aapki aawaz saaf sunai nahi di. Kripya dobara batayein."
    ),
    "BUSY_CALLBACK": (
        "Abhi seva vyast hai. Kripya kuch der baad dobara call karein."
    ),
}


class TextToSpeech(Protocol):
    """Protocol for synthesizing reply text into 8 kHz PCM16 audio frames."""

    async def synthesize(self, text: str) -> AsyncIterator[bytes]: ...


class FixedScriptCache:
    """In-memory audio byte cache for static safety and greeting prompts."""

    def __init__(self, inner: TextToSpeech) -> None:
        self._inner = inner
        self._cache: dict[str, list[bytes]] = {}

    @property
    def cached_texts(self) -> set[str]:
        return set(self._cache.keys())

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if text in self._cache:
            for frame in self._cache[text]:
                yield frame
            return

        collected: list[bytes] = []
        async for frame in self._inner.synthesize(text):
            collected.append(frame)
            yield frame

        # Cache fixed prompts or repeated readbacks
        if text in FIXED_SCRIPTS.values() or len(self._cache) < 64:
            self._cache[text] = collected


class FakeTTS:
    """Deterministic TTS stub yielding 1024-byte 8 kHz PCM16 silence frames."""

    def __init__(self, frames_per_reply: int = 2) -> None:
        self.frames_per_reply = frames_per_reply
        self.spoken_texts: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.spoken_texts.append(text)
        silence_frame = b"\x00" * FRAME_BYTES
        for _ in range(self.frames_per_reply):
            yield silence_frame

