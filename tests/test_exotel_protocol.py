from __future__ import annotations

import base64
import json

import pytest

from asha_vani.voice.exotel import (
    ExotelMediaChunker,
    ExotelProtocolError,
    build_exotel_stream_url,
    decode_exotel_media,
    parse_exotel_message,
    verify_exotel_basic_auth,
)


def test_parse_and_decode_exotel_8khz_pcm_media() -> None:
    audio = b"\x01\x00" * 1600
    message = parse_exotel_message(json.dumps({
        "event": "media", "stream_sid": "stream-1",
        "media": {"payload": base64.b64encode(audio).decode("ascii")},
    }))
    assert decode_exotel_media(message) == audio


def test_invalid_base64_or_odd_pcm_payload_is_rejected() -> None:
    with pytest.raises(ExotelProtocolError):
        decode_exotel_media({"media": {"payload": "***"}})
    with pytest.raises(ExotelProtocolError):
        decode_exotel_media({"media": {"payload": base64.b64encode(b"\x00").decode("ascii")}})


@pytest.mark.parametrize("raw", ["not-json", "[]", '{"event":"wat"}', '{"event":"media"}'])
def test_malformed_agentstream_message_is_rejected(raw: str) -> None:
    with pytest.raises(ExotelProtocolError):
        parse_exotel_message(raw)


def test_chunker_emits_provider_minimum_and_flushes_short_tail() -> None:
    chunker = ExotelMediaChunker("stream-1")
    first = chunker.push(b"\x00" * 4000)
    tail = chunker.push(b"\x00" * 500, final=True)
    assert [len(decode_exotel_media(packet)) for packet in first + tail] == [3200, 3200]
    assert all(packet["stream_sid"] == "stream-1" for packet in first + tail)
    assert all(set(packet) == {"event", "stream_sid", "media"} for packet in first + tail)
    assert all(set(packet["media"]) == {"payload"} for packet in first + tail)


def test_exotel_basic_auth_is_checked_without_echoing_credentials() -> None:
    token = base64.b64encode(b"key:secret").decode("ascii")
    assert verify_exotel_basic_auth(f"Basic {token}", "key", "secret")
    assert not verify_exotel_basic_auth(f"Basic {token}", "key", "wrong")
    assert not verify_exotel_basic_auth(None, "key", "secret")


def test_public_base_url_builds_secure_exotel_websocket_path() -> None:
    assert build_exotel_stream_url("https://demo.example/prefix/") == "wss://demo.example/prefix/ws/exotel"
    with pytest.raises(ValueError):
        build_exotel_stream_url("https://key:secret@demo.example")
