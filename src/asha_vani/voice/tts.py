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


class GnaniTimbreTTS:
    """Live Gnani Timbre TTS client (`gnani.tts.GnaniTTSRealtimeClient` / `GnaniTTSClient`).

    Follows the official `gnani-vachana` SDK rules:
    - ALWAYS passes `model="timbre-v2.5"` explicitly (SDK default `timbre-v2.0` is retired and 400s).
    - Defaults to `voice="Nalini"`, `language="hi-IN"` (or `voice="Poorvi"`, `language="auto"` for Hinglish).
    - Uses `AudioConfig(sample_rate=8000, encoding="linear_pcm", container="raw")` when streaming
      raw PCM frames, or `container="wav"` when returning a single playable browser WAV file.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice: str = "Nalini",
        model: str = "timbre-v2.5",
        language: str = "hi-IN",
        speed: float = 1.0,
        sample_rate: int = 8000,
        container: str = "raw",
    ) -> None:
        if model == "timbre-v2.0":
            raise ValueError("timbre-v2.0 is retired by Gnani API; use 'timbre-v2.5' or 'timbre-v3.0'.")
        self.api_key = api_key
        self.voice = voice
        self.model = model
        self.language = language
        self.speed = speed
        self.sample_rate = sample_rate
        self.container = container

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Stream synthesized audio chunks over WebSocket (`GnaniTTSRealtimeClient`)."""
        from gnani.tts import AudioConfig, GnaniTTSRealtimeClient

        audio_cfg = AudioConfig(
            sample_rate=self.sample_rate,
            encoding="linear_pcm",
            num_channels=1,
            sample_width=2,
            container=self.container,
        )
        client_kwargs = {"api_key": self.api_key} if self.api_key else {}
        async with GnaniTTSRealtimeClient(**client_kwargs) as client:
            async for chunk in client.synthesize(
                text,
                voice=self.voice,
                model=self.model,
                language=self.language,
                speed=self.speed,
                audio_config=audio_cfg,
            ):
                if chunk:
                    yield chunk

    def synthesize_wav_bytes(self, text: str, sample_rate: int = 24000) -> bytes:
        """Synchronous REST helper (`GnaniTTSClient`) returning a complete `.wav` byte buffer for browser `<audio>` playback."""
        from gnani.tts import AudioConfig, GnaniTTSClient

        audio_cfg = AudioConfig(
            sample_rate=sample_rate,
            encoding="linear_pcm",
            num_channels=1,
            sample_width=2,
            container="wav",
        )
        client = GnaniTTSClient(api_key=self.api_key) if self.api_key else GnaniTTSClient()
        return client.synthesize(
            text,
            voice=self.voice,
            model=self.model,
            language=self.language,
            speed=self.speed,
            audio_config=audio_cfg,
        )

