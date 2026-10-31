"""FastAPI server for ASHA-Vani: serves `/simulator`, `/ws/call`, `/ws/console`, and `/api/turn`.

Lets you open `http://127.0.0.1:8000/simulator`, press CALL on the Nokia simulator,
speak into your laptop microphone, run through `CallSession` -> `TurnRunner` -> Person B's
`call_graph`, hear the spoken reply, and watch the PHC Register table + Step-by-Step
Pipeline Inspector update in real time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from asha_vani.voice.events import CallEvent
from asha_vani.voice.session import CallSession
from asha_vani.voice.stt import GnaniPrismaSTT
from asha_vani.voice.tts import FakeTTS, FixedScriptCache, GnaniTimbreTTS, TextToSpeech
from asha_vani.voice.turn_runner import LangGraphTurnRunner

load_dotenv()
_SIMULATOR_HTML_PATH = Path(__file__).resolve().parent / "voice" / "simulator.html"

# In-memory WAV cache so identical prompts (greeting, follow-ups, repeats) cost 0 extra Gnani credits
_TTS_WAV_CACHE: dict[str, bytes] = {}
INITIAL_GREETING_HINGLISH = (
    "Namaste Didi! ASHA-Vani mein aapka swagat hai. "
    "Aap kis gaon mein hain aur kis marij ke ghar se bol rahi hain?"
)

app = FastAPI(title="ASHA-Vani Voice & PHC Server", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class WebSocketConsoleBroadcaster:
    """EventSink that broadcasts every CallEvent (`call_started`, `turn_completed`, `call_ended`) to connected PHC consoles."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.history: list[dict[str, Any]] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)
        for past_event in self.history[-50:]:
            try:
                await ws.send_json(past_event)
            except Exception:
                pass

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def emit(self, event: CallEvent) -> None:
        data = event.model_dump(mode="json")
        self.history.append(data)
        dead: list[WebSocket] = []
        for ws in list(self.clients):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


broadcaster = WebSocketConsoleBroadcaster()
shared_turn_runner = LangGraphTurnRunner(require_full_checklist=True)
active_sessions: dict[str, CallSession] = {}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "gnani_configured": bool(os.getenv("GNANI_API_KEY", "").strip()),
        "active_calls": len(active_sessions),
        "console_clients": len(broadcaster.clients),
    }


@app.get("/api/tts")
def synthesize_tts_wav(
    text: str,
    voice: str | None = None,
    language: str | None = None,
) -> Response:
    """Synthesize a WAV audio clip using Gnani Timbre v2.5 (`GnaniTTSClient`) with in-memory caching."""
    api_key = os.getenv("GNANI_API_KEY", "").strip()
    if not api_key:
        return Response(content=b"", status_code=204)

    # Default to Poorvi + auto for Roman Hinglish, or respect .env override
    chosen_voice = voice or os.getenv("GNANI_TTS_VOICE", "Poorvi")
    chosen_lang = language or os.getenv("GNANI_TTS_LANG", "auto")
    # If .env has Nalini + hi-IN, Poorvi/auto handles Roman Hinglish much more naturally;
    # keep user choice if explicitly set, and cache by (voice, lang, text)
    cache_key = f"{chosen_voice}:{chosen_lang}:{text.strip()}"
    if cache_key in _TTS_WAV_CACHE:
        return Response(content=_TTS_WAV_CACHE[cache_key], media_type="audio/wav")

    try:
        tts = GnaniTimbreTTS(
            api_key=api_key,
            voice=chosen_voice,
            model="timbre-v2.5",
            language=chosen_lang,
        )
        wav_bytes = tts.synthesize_wav_bytes(text)
    except Exception:
        # Fallback to Nalini + hi-IN if voice/lang pair was rejected
        tts = GnaniTimbreTTS(api_key=api_key, voice="Nalini", model="timbre-v2.5", language="hi-IN")
        wav_bytes = tts.synthesize_wav_bytes(text)

    if len(_TTS_WAV_CACHE) < 128:
        _TTS_WAV_CACHE[cache_key] = wav_bytes
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/", response_class=HTMLResponse)
@app.get("/simulator", response_class=HTMLResponse)
async def serve_simulator() -> HTMLResponse:
    """Serve the 75% PHC Register + 25% Nokia Feature-Phone Simulator workspace."""
    if _SIMULATOR_HTML_PATH.exists():
        return HTMLResponse(_SIMULATOR_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Simulator HTML not found in src/asha_vani/voice/simulator.html</h1>", status_code=404)


class UtteranceRequest(BaseModel):
    session_id: str | None = None
    caller: str = "+919900000001"
    transcript: str
    stt_latency_ms: float = 120.0


@app.post("/api/turn")
async def post_utterance_turn(req: UtteranceRequest) -> dict[str, Any]:
    """Run one spoken utterance through `CallSession` -> `TurnRunner` -> `call_graph` and broadcast to `/ws/console`."""
    sid = req.session_id or f"call-{uuid.uuid4().hex[:8]}"
    session = active_sessions.get(sid)
    if session is None:
        session = CallSession(
            sid,
            caller=req.caller,
            turn_runner=shared_turn_runner,
            tts=FakeTTS(),
            sinks=[broadcaster],
        )
        active_sessions[sid] = session
        await session.start()

    out = await session.process_utterance(req.transcript, stt_latency_ms=req.stt_latency_ms)
    if out is None:
        return {"session_id": sid, "ok": False, "reason": "empty_utterance"}

    return {
        "session_id": sid,
        "ok": True,
        "turn": out.model_dump(mode="json"),
    }


@app.post("/api/call/{session_id}/end")
async def end_call_endpoint(session_id: str, reason: str = "COMPLETED") -> dict[str, Any]:
    session = active_sessions.pop(session_id, None)
    if session is None:
        return {"session_id": session_id, "ended": False}
    ev = await session.end(reason=reason)
    return {"session_id": session_id, "ended": True, "event": ev.model_dump(mode="json")}


@app.websocket("/ws/console")
async def ws_console(ws: WebSocket) -> None:
    """Live turn-level event stream for the Left 75% PHC Register table & drawer."""
    await broadcaster.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(ws)


@app.websocket("/ws/call")
async def ws_call(ws: WebSocket, session_id: str | None = None, caller: str = "+919900000001") -> None:
    """Live call WebSocket for the Right 25% Nokia Simulator.

    Accepts:
    - Binary frames: 8 kHz 16-bit mono PCM audio (passed through `FrameBuffer` & half-duplex gate)
    - JSON text frames: `{"type": "utterance", "text": "...", "stt_latency_ms": 135}` or `{"type": "hangup"}`
    """
    await ws.accept()
    sid = session_id or f"call-{uuid.uuid4().hex[:8]}"
    session = CallSession(
        sid,
        caller=caller,
        turn_runner=shared_turn_runner,
        tts=FakeTTS(),
        sinks=[broadcaster],
    )
    active_sessions[sid] = session
    await session.start()
    await ws.send_json(
        {
            "type": "call_started",
            "session_id": sid,
            "greeting_text": INITIAL_GREETING_HINGLISH,
        }
    )

    end_reason = "DROPPED"
    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes") is not None:
                # Condition & frame raw 8kHz PCM16 audio into exact 1024-byte frames
                session.push_mic_audio(msg["bytes"])
            elif msg.get("text") is not None:
                import json

                data = json.loads(msg["text"])
                mtype = data.get("type")
                if mtype == "utterance":
                    out = await session.process_utterance(
                        str(data.get("text", "")),
                        stt_latency_ms=float(data.get("stt_latency_ms", 120.0)),
                    )
                    if out is not None:
                        await ws.send_json(
                            {
                                "type": "turn_reply",
                                "session_id": sid,
                                "reply_text": out.reply_text,
                                "turn": out.model_dump(mode="json"),
                            }
                        )
                elif mtype == "hangup":
                    end_reason = "COMPLETED"
                    break
    except WebSocketDisconnect:
        pass
    finally:
        active_sessions.pop(sid, None)
        await session.end(reason=end_reason)

