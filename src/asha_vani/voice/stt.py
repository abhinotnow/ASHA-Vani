"""Streaming Speech-to-Text (STT) interface, End-of-Utterance detector, and Fakes.

Enforces AGENTS.md 4.2 Edge-Case Invariant #2:
- Finish turn immediately if the ASHA ends with the trigger word `"over"` (and strip `"over"`).
- Otherwise finish turn on natural pause (~1.2s silence / Prisma final utterance event).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_PAUSE_TIMEOUT_S: float = 1.2

# Matches trailing "over" (or Hindi "ओवर") at the end of a spoken turn
_TRAILING_OVER_RE = re.compile(
    r"(?:\s+|\b)(?:over|ओवर)[\s\.,!?;:]*$",
    re.IGNORECASE,
)


def strip_trailing_over(text: str) -> tuple[str, bool]:
    """Check if `text` ends with the trigger word 'over'; return (cleaned_text, triggered)."""
    stripped = text.strip()
    if not stripped:
        return "", False
    match = _TRAILING_OVER_RE.search(stripped)
    if match:
        cleaned = stripped[: match.start()].strip(" ,.;:!?-")
        return cleaned, True
    return stripped, False


@dataclass(frozen=True)
class STTChunk:
    """One event from a streaming STT engine (Gnani Prisma or FakeSTT)."""

    text: str
    is_final: bool = False
    latency_ms: float = 0.0
    segment_id: str | None = None
    detected_language_code: str | None = None


class EndOfUtteranceDetector:
    """Accumulates STT chunks and yields complete, cleaned utterances on 'over' or VAD speech_end."""

    def __init__(self, pause_timeout_s: float = DEFAULT_PAUSE_TIMEOUT_S) -> None:
        self.pause_timeout_s = pause_timeout_s
        self._final_segments: list[str] = []
        self._latest_partial: str = ""

    @property
    def has_pending_speech(self) -> bool:
        """True if there is buffered partial or final speech waiting for VAD end."""
        return bool(self._latest_partial or self._final_segments)

    def feed(self, chunk: STTChunk) -> str | None:
        """Feed a partial or final STT chunk. Returns a completed utterance string or None."""
        text = chunk.text.strip()
        if not text:
            if chunk.is_final and self.has_pending_speech:
                return self.flush()
            return None

        cleaned, has_over = strip_trailing_over(text)
        if has_over:
            self._latest_partial = cleaned
            return self.flush()

        if chunk.is_final:
            self._latest_partial = ""
            self._final_segments.append(cleaned)
            return self.flush()

        # Partial transcript without "over": buffer partial preview, wait for VAD speech_end
        self._latest_partial = cleaned
        return None

    def flush(self) -> str | None:
        """Flush buffered text on silence timeout or end of stream."""
        parts = list(self._final_segments)
        if self._latest_partial:
            parts.append(self._latest_partial)
        self._final_segments.clear()
        self._latest_partial = ""
        if not parts:
            return None
        full = " ".join(p for p in parts if p).strip()
        cleaned, _ = strip_trailing_over(full)
        return cleaned or None

    def clear(self) -> None:
        """Discard partial text (used when half-duplex mutes the mic)."""
        self._final_segments.clear()
        self._latest_partial = ""


class SpeechToText(Protocol):
    """Protocol for streaming 8 kHz PCM16 1024-byte frames into STTChunks."""

    async def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[STTChunk]: ...


class FakeSTT:
    """Deterministic STT stub for unit tests, replay scripts, and load testing."""

    def __init__(self, chunks: list[STTChunk | str]) -> None:
        self._chunks: list[STTChunk] = [
            c if isinstance(c, STTChunk) else STTChunk(text=c, is_final=True)
            for c in chunks
        ]

    async def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[STTChunk]:
        # Consume at least one frame if available, then yield scripted chunks
        async for _ in frames:
            break
        for chunk in self._chunks:
            yield chunk


class GnaniPrismaSTT:
    """Live Gnani Prisma Streaming STT client (`gnani.stt.GnaniSTTStreamClient`).

    Uses the official `gnani-vachana` SDK:
    - Streams 8 kHz (or 16 kHz) 16-bit mono PCM in exact 1,024-byte frames (`STREAM_CHUNK_BYTES = 1024`).
    - Defaults to `language_code="hi-IN"` and `format="transcribe"` (enables ITN so spoken numbers
      come back as digits like `150`, `95`, `6.5`).
    - Runs sending and receiving as concurrent asyncio tasks (per Gnani SDK docs).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        language_code: str = "hi-IN",
        language_candidates: list[str] | None = None,
        sample_rate: int = 8000,
        format: str = "transcribe",
        itn_native_numerals: bool = False,
        base_url: str = "https://api.vachana.ai",
        realtime_pace: bool = False,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.language_code = language_code
        self.language_candidates = language_candidates
        self.sample_rate = sample_rate
        self.format = format
        self.itn_native_numerals = itn_native_numerals
        self.base_url = base_url
        self.realtime_pace = realtime_pace
        self.model = model

    async def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[STTChunk]:
        """Stream 1024-byte PCM16 frames to Gnani Prisma and yield `STTChunk` events."""
        import asyncio
        from gnani.stt import (
            GnaniSTTStreamClient,
            StreamErrorEvent,
            StreamProcessingEvent,
            StreamTranscriptEvent,
        )

        resolved_model = self.model
        if not resolved_model and (self.language_code == "indic-auto" or bool(self.language_candidates)):
            resolved_model = "prisma-v3.0"

        kwargs: dict[str, object] = {
            "language_code": self.language_code,
            "sample_rate": self.sample_rate,
            "format": self.format,
            "itn_native_numerals": self.itn_native_numerals,
            "base_url": self.base_url,
        }
        if resolved_model:
            kwargs["model"] = resolved_model
        if self.language_candidates:
            kwargs["language_candidates"] = list(self.language_candidates)
        if self.api_key:
            kwargs["api_key"] = self.api_key

        frame_sleep_s = (1024 / 2 / self.sample_rate) if self.realtime_pace else 0.0

        async with GnaniSTTStreamClient(**kwargs) as client:  # type: ignore[arg-type]
            queue: asyncio.Queue[STTChunk | None] = asyncio.Queue()

            async def _sender() -> None:
                try:
                    async for frame in frames:
                        if not frame:
                            continue
                        await client.send_audio(frame)
                        if frame_sleep_s > 0.0:
                            await asyncio.sleep(frame_sleep_s)
                finally:
                    await client.close()

            async def _receiver() -> None:
                try:
                    async for event in client:
                        if isinstance(event, StreamTranscriptEvent):
                            cleaned, has_over = strip_trailing_over(event.text)
                            raw = getattr(event, "raw", {}) or {}
                            is_final = bool(has_over or raw.get("is_final") or raw.get("final"))
                            logger.info(
                                "Prisma STT transcript: '%s' (is_final=%s, lang=%s)",
                                event.text,
                                is_final,
                                getattr(event, "detected_language_code", None),
                            )
                            await queue.put(
                                STTChunk(
                                    text=event.text,
                                    is_final=is_final,
                                    latency_ms=float(getattr(event, "latency", 0) or 0.0),
                                    segment_id=getattr(event, "segment_id", None),
                                    detected_language_code=getattr(event, "detected_language_code", None),
                                )
                            )
                        elif isinstance(event, StreamProcessingEvent):
                            raw = getattr(event, "raw", {}) or {}
                            msg_type = raw.get("type", "")
                            if msg_type in ("speech_end", "vad_end"):
                                logger.info("Prisma VAD: %s received -> user finished speaking", msg_type)
                                await queue.put(
                                    STTChunk(
                                        text="",
                                        is_final=True,
                                    )
                                )
                        elif isinstance(event, StreamErrorEvent):
                            logger.error(
                                "Gnani Prisma StreamErrorEvent: %s",
                                getattr(event, "message", str(event)),
                            )
                            break
                finally:
                    await queue.put(None)

            send_task = asyncio.create_task(_sender())
            recv_task = asyncio.create_task(_receiver())
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item
            finally:
                await asyncio.gather(send_task, recv_task, return_exceptions=True)

    def transcribe_wav_bytes(self, audio_bytes: bytes, filename: str = "utterance.wav") -> dict[str, object]:
        """REST helper (`POST /stt/v3`) for short pre-recorded or buffered utterance clips (<30s)."""
        from gnani.stt import GnaniSTTClient

        client = GnaniSTTClient(api_key=self.api_key) if self.api_key else GnaniSTTClient()
        return client.transcribe_bytes(
            audio_bytes,
            filename=filename,
            language_code=self.language_code,
            format=self.format,
            itn_native_numerals=self.itn_native_numerals,
        )
