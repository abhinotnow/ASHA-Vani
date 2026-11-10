"""Small, testable parts of the Exotel AgentStream WebSocket protocol."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import asyncio
import logging
import os
from urllib.parse import urlsplit
from typing import Any

EXOTEL_MIN_MEDIA_BYTES = 3200
EXOTEL_MEDIA_ALIGNMENT = 320
EXOTEL_MAX_MEDIA_BYTES = 100_000
logger = logging.getLogger(__name__)


def build_exotel_stream_url(public_base_url: str) -> str:
    """Build the AgentStream WSS endpoint from the configured public HTTPS URL."""
    parsed = urlsplit(public_base_url.strip())
    if parsed.scheme not in {"https", "http"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("PUBLIC_BASE_URL must be a public HTTP(S) origin without credentials")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/")
    return f"{scheme}://{parsed.netloc}{path}/ws/exotel"


class ExotelProtocolError(ValueError):
    """The provider sent an unsupported or malformed stream message."""


def parse_exotel_message(text: str) -> dict[str, Any]:
    try:
        message = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExotelProtocolError("message is not valid JSON") from exc
    if not isinstance(message, dict):
        raise ExotelProtocolError("message must be a JSON object")
    event = message.get("event")
    if event not in {"connected", "start", "media", "dtmf", "stop", "mark", "clear"}:
        raise ExotelProtocolError("unsupported event")
    if event in {"start", "media", "stop", "mark", "clear"} and not message.get("stream_sid"):
        raise ExotelProtocolError("stream_sid is required")
    return message


def decode_exotel_media(message: dict[str, Any]) -> bytes:
    media = message.get("media")
    payload = media.get("payload") if isinstance(media, dict) else None
    if not isinstance(payload, str) or not payload:
        raise ExotelProtocolError("media payload is missing")
    try:
        audio = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ExotelProtocolError("media payload is not valid base64") from exc
    if not audio or len(audio) > EXOTEL_MAX_MEDIA_BYTES or len(audio) % 2:
        raise ExotelProtocolError("media payload has an invalid PCM16 size")
    return audio


class ExotelMediaChunker:
    """Buffers raw PCM16 and emits provider-sized 8 kHz media packets."""

    def __init__(self, stream_sid: str) -> None:
        self.stream_sid = stream_sid
        self._buffer = bytearray()

    def push(self, pcm16: bytes, *, final: bool = False) -> list[dict[str, Any]]:
        if len(pcm16) % 2:
            raise ExotelProtocolError("PCM16 input must contain whole samples")
        self._buffer.extend(pcm16)
        output: list[dict[str, Any]] = []
        while len(self._buffer) >= EXOTEL_MIN_MEDIA_BYTES:
            size = EXOTEL_MIN_MEDIA_BYTES
            chunk = bytes(self._buffer[:size])
            del self._buffer[:size]
            output.append(self._message(chunk))
        if final and self._buffer:
            # Exotel documents a 3.2 kB minimum. Pad only the last packet with
            # silence so short tails are still delivered and the call can end.
            chunk = bytes(self._buffer)
            self._buffer.clear()
            if len(chunk) < EXOTEL_MIN_MEDIA_BYTES:
                chunk += b"\x00" * (EXOTEL_MIN_MEDIA_BYTES - len(chunk))
            remainder = len(chunk) % EXOTEL_MEDIA_ALIGNMENT
            if remainder:
                chunk += b"\x00" * (EXOTEL_MEDIA_ALIGNMENT - remainder)
            output.append(self._message(chunk))
        return output

    def _message(self, pcm16: bytes) -> dict[str, Any]:
        return {
            "event": "media",
            "stream_sid": self.stream_sid,
            "media": {"payload": base64.b64encode(pcm16).decode("ascii")},
        }


class ExotelTTSStream:
    """Adapt Timbre raw PCM to Exotel media packets and wait for playback marks."""

    def __init__(self, websocket: Any, stream_sid: str) -> None:
        self.websocket = websocket
        self.chunker = ExotelMediaChunker(stream_sid)
        self._marks: dict[str, asyncio.Future[None]] = {}
        self._mark_seq = 0
        self._tts = None

    def acknowledge_mark(self, message: dict[str, Any]) -> None:
        name = (message.get("mark") or {}).get("name")
        future = self._marks.pop(str(name), None)
        if future is not None and not future.done():
            future.set_result(None)

    async def synthesize(self, text: str):
        from asha_vani.voice.tts import GnaniTimbreTTS

        if not os.getenv("GNANI_API_KEY", "").strip():
            raise RuntimeError("Gnani TTS is not configured")
        if self._tts is None:
            self._tts = GnaniTimbreTTS(
                api_key=os.getenv("GNANI_API_KEY", "").strip(),
                voice=os.getenv("GNANI_TTS_VOICE", "Poorvi"),
                model="timbre-v2.5",
                language=os.getenv("GNANI_TTS_LANG", "auto"),
                sample_rate=8000,
                container="raw",
            )
        sent_packets = 0
        sent_audio_bytes = 0
        async for pcm in self._tts.synthesize(text):
            for message in self.chunker.push(pcm):
                await self.websocket.send_json(message)
                sent_packets += 1
                sent_audio_bytes += len(base64.b64decode(message["media"]["payload"]))
                # Exotel's outbound media is consumed as a live stream; pace 200ms packets.
                await asyncio.sleep(0.2)
            yield pcm
        for message in self.chunker.push(b"", final=True):
            await self.websocket.send_json(message)
            sent_packets += 1
            sent_audio_bytes += len(base64.b64decode(message["media"]["payload"]))
            await asyncio.sleep(0.2)
        logger.info("Exotel TTS sent %d media packets (%d PCM bytes)", sent_packets, sent_audio_bytes)
        self._mark_seq += 1
        name = f"asha-vani-tts-{self._mark_seq}"
        future = asyncio.get_running_loop().create_future()
        self._marks[name] = future
        await self.websocket.send_json({
            "event": "mark", "stream_sid": self.chunker.stream_sid,
            "mark": {"name": name},
        })
        try:
            await asyncio.wait_for(future, timeout=15.0)
        finally:
            self._marks.pop(name, None)


def verify_exotel_basic_auth(
    authorization: str | None,
    api_key: str,
    api_token: str,
) -> bool:
    """Validate Exotel's documented Basic Auth header without logging secrets."""
    if not authorization or not api_key or not api_token:
        return False
    scheme, separator, encoded = authorization.partition(" ")
    if not separator or scheme.casefold() != "basic":
        return False
    try:
        supplied = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False
    expected = f"{api_key}:{api_token}"
    return hmac.compare_digest(supplied, expected)
