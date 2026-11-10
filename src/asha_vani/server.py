"""FastAPI server for ASHA-Vani: serves `/simulator`, `/ws/call`, `/ws/console`, and `/api/turn`.

Lets you open `http://127.0.0.1:8000/simulator`, press CALL on the Nokia simulator,
speak into your laptop microphone, run through `CallSession` -> `TurnRunner` -> Person B's
`call_graph`, hear the spoken reply, and watch the PHC Register table + Step-by-Step
Pipeline Inspector update in real time.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any
import urllib.request
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
from asha_vani.persistence import SupabaseEventSink
from asha_vani.voice.audio import FRAME_BYTES
from asha_vani.voice.exotel import (
    ExotelMediaChunker,
    ExotelProtocolError,
    ExotelTTSStream,
    build_exotel_stream_url,
    decode_exotel_media,
    parse_exotel_message,
    verify_exotel_basic_auth,
)

load_dotenv()
logger = logging.getLogger(__name__)
_SIMULATOR_HTML_PATH = Path(__file__).resolve().parent / "voice" / "simulator.html"
_SIMULATOR_FURROW_HTML_PATH = Path(__file__).resolve().parent / "voice" / "simulator_furrow.html"
_SOLUTION_PITCH_HTML_PATH = Path(__file__).resolve().parents[2] / "docs" / "solution_pitch.html"

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

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def emit(self, event: CallEvent) -> None:
        data = event.model_dump(mode="json")
        self.history.append(data)
        await self.broadcast_raw(data)

    async def broadcast_raw(self, data: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in list(self.clients):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


broadcaster = WebSocketConsoleBroadcaster()
active_sessions: dict[str, CallSession] = {}
dispatches_store: dict[str, dict[str, Any]] = {}
supabase_sink: SupabaseEventSink | None = None
if os.getenv("SUPABASE_URL", "").strip() and os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
    try:
        supabase_sink = SupabaseEventSink(
            os.environ["SUPABASE_URL"].strip(),
            os.environ["SUPABASE_SERVICE_ROLE_KEY"].strip(),
        )
    except Exception as exc:
        # The text/voice demo can still run without persistence credentials.
        import logging

        logging.getLogger(__name__).exception("Supabase persistence is disabled: %s", exc)

shared_turn_runner = LangGraphTurnRunner(
    require_full_checklist=True,
    history_lookup=supabase_sink.find_patient_history if supabase_sink is not None else None,
)


def _session_sinks() -> list[Any]:
    sinks: list[Any] = [broadcaster]
    if supabase_sink is not None:
        sinks.append(supabase_sink)
    return sinks


class WebSocketTimbreStream:
    """Stream raw 8 kHz PCM from Timbre to the active browser call socket."""

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        self._client = GnaniTimbreTTS(
            api_key=os.getenv("GNANI_API_KEY", "").strip() or None,
            voice=os.getenv("GNANI_TTS_VOICE", "Poorvi"),
            model="timbre-v2.5",
            language=os.getenv("GNANI_TTS_LANG", "auto"),
            sample_rate=8000,
            container="raw",
        )

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if not os.getenv("GNANI_API_KEY", "").strip():
            await self._ws.send_json({"type": "tts_error", "message": "GNANI_API_KEY is not configured", "text": text})
            return
        await self._ws.send_json({"type": "tts_start", "sample_rate": 8000})
        try:
            async for chunk in self._client.synthesize(text):
                await self._ws.send_bytes(chunk)
                yield chunk
            await self._ws.send_json({"type": "tts_end"})
        except Exception as exc:
            await self._ws.send_json({"type": "tts_error", "message": str(exc)[:200], "text": text})


@app.get("/health")
async def health() -> dict[str, Any]:
    public_base_url = os.getenv("PUBLIC_BASE_URL", "").strip()
    try:
        exotel_stream_url = build_exotel_stream_url(public_base_url) if public_base_url else None
    except ValueError:
        exotel_stream_url = None
    return {
        "ok": True,
        "gnani_configured": bool(os.getenv("GNANI_API_KEY", "").strip()),
        "active_calls": len(active_sessions),
        "console_clients": len(broadcaster.clients),
        "supabase_configured": supabase_sink is not None,
        "exotel_configured": all(os.getenv(key, "").strip() for key in (
            "EXOTEL_API_KEY", "EXOTEL_API_TOKEN", "EXOTEL_ACCOUNT_SID",
            "EXOTEL_EXOPHONE", "PUBLIC_BASE_URL",
        )),
        "exotel_stream_url": exotel_stream_url,
    }


@app.get("/api/register")
async def load_register() -> dict[str, Any]:
    """Return saved final visit reports and call history for console hydration."""
    if supabase_sink is None:
        return {"configured": False, "visits": [], "events": []}
    try:
        data = await supabase_sink.load_register()
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Saved register could not be loaded")
        return {"configured": True, "available": False, "visits": [], "events": [], "calls": []}
    return {"configured": True, "available": True, **data}


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


@app.get("/furrow", response_class=HTMLResponse)
@app.get("/simulator-furrow", response_class=HTMLResponse)
async def serve_simulator_furrow() -> HTMLResponse:
    """Serve the Hallmark Furrow edition of the ASHA-Vani Field Register & Nokia Simulator workspace."""
    if _SIMULATOR_FURROW_HTML_PATH.exists():
        return HTMLResponse(_SIMULATOR_FURROW_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>Furrow Simulator HTML not found in src/asha_vani/voice/simulator_furrow.html</h1>",
        status_code=404,
    )


@app.get("/pitch", response_class=HTMLResponse)
async def serve_solution_pitch() -> HTMLResponse:
    """Serve the 1-Page A4 Solution Pitch document."""
    if _SOLUTION_PITCH_HTML_PATH.exists():
        return HTMLResponse(_SOLUTION_PITCH_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Solution pitch HTML not found in docs/solution_pitch.html</h1>", status_code=404)


class OfficerLoginRequest(BaseModel):
    email: str = "dr.verma@asha-vani.in"
    password: str = ""
    mode: str = "demo"


@app.post("/api/auth/login")
async def officer_login(req: OfficerLoginRequest) -> dict[str, Any]:
    """Authenticate Medical Officer via Supabase Auth REST API when configured, or issue Demo Officer session."""
    base_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    api_key = (
        os.getenv("SUPABASE_ANON_KEY", "").strip()
        or os.getenv("SUPABASE_KEY", "").strip()
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )

    if req.mode != "demo" and base_url and api_key and req.password:
        auth_url = f"{base_url}/auth/v1/token?grant_type=password"
        payload = json.dumps({"email": req.email, "password": req.password}).encode("utf-8")
        http_req = urllib.request.Request(
            auth_url,
            data=payload,
            method="POST",
            headers={
                "apikey": api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(http_req, timeout=5.0) as resp:
                auth_data = json.loads(resp.read().decode("utf-8"))
                return {
                    "ok": True,
                    "provider": "supabase",
                    "officer_name": req.email.split("@")[0].replace(".", " ").title(),
                    "email": req.email,
                    "role": "Medical Officer · Triage & Dispatch",
                    "access_token": auth_data.get("access_token", ""),
                }
        except Exception as exc:
            return {"ok": False, "error": f"Supabase Auth failed: {exc}"}

    return {
        "ok": True,
        "provider": "supabase_demo" if supabase_sink is not None else "local_session",
        "officer_name": "Dr. A. Verma",
        "email": req.email or "dr.verma@asha-vani.in",
        "role": "Medical Officer · Triage & Dispatch",
    }


class DispatchRequest(BaseModel):
    visit_id: str
    unit_type: str  # "108_UNIT" | "ANM_MOBILE" | "HOME_VISIT"
    eta_minutes: int = 14
    notes: str = ""


@app.post("/api/dispatch")
async def create_dispatch(req: DispatchRequest) -> dict[str, Any]:
    """Record a Home Dispatch order for a visit and broadcast to all connected consoles."""
    now_iso = datetime.now(timezone.utc).isoformat()
    unit_labels = {
        "108_UNIT": f"108 Unit · {req.eta_minutes}m ETA",
        "ANM_MOBILE": f"ANM Mobile · {req.eta_minutes}m ETA",
        "HOME_VISIT": "Home Follow-up Logged",
    }
    record = {
        "visit_id": req.visit_id,
        "unit_type": req.unit_type,
        "label": unit_labels.get(req.unit_type, req.unit_type),
        "eta_minutes": req.eta_minutes,
        "dispatched_at": now_iso,
        "notes": req.notes,
    }
    dispatches_store[req.visit_id] = record
    await broadcaster.broadcast_raw({"event_type": "dispatch_updated", "payload": record})
    return {"ok": True, "dispatch": record}


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
            sinks=_session_sinks(),
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
    """Accept 8 kHz PCM16 for Prisma plus JSON utterances for replay/manual mode."""
    await ws.accept()
    sid = session_id or f"call-{uuid.uuid4().hex[:8]}"
    session = CallSession(
        sid,
        caller=caller,
        turn_runner=shared_turn_runner,
        tts=WebSocketTimbreStream(ws),
        sinks=_session_sinks(),
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
    async for _ in WebSocketTimbreStream(ws).synthesize(INITIAL_GREETING_HINGLISH):
        pass

    end_reason = "DROPPED"
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=100)

    async def frames() -> AsyncIterator[bytes]:
        while True:
            frame = await audio_queue.get()
            if frame is None:
                return
            yield frame

    async def consume_prisma() -> None:
        try:
            prisma = GnaniPrismaSTT(
                api_key=os.getenv("GNANI_API_KEY", "").strip() or None,
                language_code=os.getenv("GNANI_STT_LANGUAGE", "hi-IN"),
                sample_rate=8000,
                format="transcribe",
            )
            async for chunk in prisma.stream(frames()):
                turn = await session.receive_stt_chunk(chunk, stt_latency_ms=chunk.latency_ms)
                if turn is not None:
                    await ws.send_json({
                        "type": "turn_reply",
                        "session_id": sid,
                        "reply_text": turn.reply_text,
                        "turn": turn.model_dump(mode="json"),
                    })
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await ws.send_json({"type": "stt_error", "message": str(exc)[:200]})

    prisma_task = asyncio.create_task(consume_prisma())
    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes") is not None:
                # FrameBuffer yields exact 1,024-byte audio blocks for Prisma.
                for frame in session.push_mic_audio(msg["bytes"]):
                    if len(frame) != FRAME_BYTES:
                        continue
                    if audio_queue.full():
                        try:
                            audio_queue.get_nowait()  # discard stale audio under overload
                        except asyncio.QueueEmpty:
                            pass
                    audio_queue.put_nowait(frame)
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
                        if out.next_action == "END_CALL":
                            end_reason = "COMPLETED"
                            await ws.send_json({"type": "call_ended", "session_id": sid, "reason": "COMPLETED"})
                            break
                elif mtype == "hangup":
                    end_reason = "COMPLETED"
                    break
    except WebSocketDisconnect:
        pass
    finally:
        if audio_queue.full():
            try:
                audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        audio_queue.put_nowait(None)
        if not prisma_task.done():
            prisma_task.cancel()
        await asyncio.gather(prisma_task, return_exceptions=True)
        active_sessions.pop(sid, None)
        await session.end(reason=end_reason)
        if end_reason == "COMPLETED":
            try:
                await ws.close(code=1000)
            except Exception:
                pass


def _exotel_start_value(start: dict[str, Any], *keys: str) -> str | None:
    for container in (start, start.get("custom_parameters", {}), start.get("customParameters", {})):
        if isinstance(container, dict):
            for key in keys:
                value = container.get(key)
                if value:
                    return str(value)
    return None


@app.websocket("/ws/exotel")
async def ws_exotel(ws: WebSocket) -> None:
    """Bridge Exotel AgentStream's JSON/base64 PCM to the shared call session."""
    if not verify_exotel_basic_auth(
        ws.headers.get("authorization"),
        os.getenv("EXOTEL_API_KEY", "").strip(),
        os.getenv("EXOTEL_API_TOKEN", "").strip(),
    ):
        await ws.close(code=4401, reason="Unauthorized")
        return
    await ws.accept()
    stream_sid: str | None = None
    session: CallSession | None = None
    tts: ExotelTTSStream | None = None
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)
    prisma_task: asyncio.Task[None] | None = None
    greeting_task: asyncio.Task[None] | None = None
    call_done = asyncio.Event()
    call_failed = asyncio.Event()
    end_reason = "DROPPED"
    inbound_media_messages = 0

    async def frames() -> AsyncIterator[bytes]:
        while True:
            frame = await audio_queue.get()
            if frame is None:
                return
            yield frame

    async def process_prisma() -> None:
        assert session is not None
        try:
            prisma = GnaniPrismaSTT(
                api_key=os.getenv("GNANI_API_KEY", "").strip() or None,
                language_code=os.getenv("GNANI_STT_LANGUAGE", "hi-IN"),
                sample_rate=8000,
                format="transcribe",
                realtime_pace=True,
            )
            async for chunk in prisma.stream(frames()):
                turn = await session.receive_stt_chunk(chunk, stt_latency_ms=chunk.latency_ms)
                if turn is not None and turn.next_action == "END_CALL":
                    # ExotelTTSStream waits for a playback mark before the turn returns.
                    call_done.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Exotel call speech processing failed")
            call_failed.set()

    async def play_exotel_greeting() -> None:
        """Play the greeting concurrently so the reader can receive its mark ack."""
        assert session is not None and tts is not None
        session.duplex_gate.start_speaking()
        session.frame_buffer.clear()
        session.eou_detector.clear()
        try:
            logger.info("Exotel greeting synthesis started")
            async for _ in tts.synthesize(INITIAL_GREETING_HINGLISH):
                pass
            logger.info("Exotel greeting synthesis and playback mark completed")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Exotel greeting playback failed")
            call_failed.set()
        finally:
            session.duplex_gate.stop_speaking()

    try:
        logger.info("Exotel AgentStream WebSocket accepted")
        while True:
            receive_task = asyncio.create_task(ws.receive_text())
            done_task = asyncio.create_task(call_done.wait())
            failed_task = asyncio.create_task(call_failed.wait())
            finished, _ = await asyncio.wait(
                {receive_task, done_task, failed_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if failed_task in finished and call_failed.is_set():
                receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)
                done_task.cancel()
                await asyncio.gather(done_task, return_exceptions=True)
                end_reason = "DROPPED"
                break
            if done_task in finished and call_done.is_set():
                receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)
                failed_task.cancel()
                await asyncio.gather(failed_task, return_exceptions=True)
                end_reason = "COMPLETED"
                break
            done_task.cancel()
            failed_task.cancel()
            await asyncio.gather(done_task, return_exceptions=True)
            await asyncio.gather(failed_task, return_exceptions=True)
            raw = receive_task.result()
            try:
                message = parse_exotel_message(raw)
            except ExotelProtocolError:
                # Avoid logging provider payloads, which may contain caller data.
                await ws.close(code=4400, reason="Malformed AgentStream message")
                break
            event = message["event"]
            if event == "connected":
                logger.info("Exotel AgentStream connected event received")
                continue
            if event == "start":
                start = message.get("start") or {}
                stream_sid = str(message.get("stream_sid") or start.get("stream_sid") or "")
                call_sid = _exotel_start_value(start, "call_sid", "CallSid", "callSid") or stream_sid
                caller = _exotel_start_value(start, "from", "From", "caller") or "UNKNOWN"
                sid = f"exotel-{call_sid}"
                tts = ExotelTTSStream(ws, stream_sid)
                session = CallSession(
                    sid, caller=caller, source="EXOTEL", turn_runner=shared_turn_runner,
                    tts=tts, sinks=_session_sinks(),
                )
                active_sessions[sid] = session
                await session.start()
                logger.info("Exotel call session started")
                prisma_task = asyncio.create_task(process_prisma())
                # Run TTS separately: this handler must keep reading mark events
                # while the greeting waits for Exotel's playback acknowledgment.
                greeting_task = asyncio.create_task(play_exotel_greeting())
                continue
            if event == "media":
                if session is None:
                    raise ExotelProtocolError("media arrived before start")
                pcm = decode_exotel_media(message)
                inbound_media_messages += 1
                if inbound_media_messages == 1:
                    logger.info("First Exotel inbound audio packet received (%d bytes)", len(pcm))
                for frame in session.push_mic_audio(pcm):
                    if audio_queue.full():
                        try:
                            audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    audio_queue.put_nowait(frame)
                if call_done.is_set():
                    end_reason = "COMPLETED"
                    break
            elif event == "mark":
                logger.info("Exotel playback mark received")
                if tts is not None:
                    tts.acknowledge_mark(message)
                if call_done.is_set():
                    end_reason = "COMPLETED"
                    break
            elif event == "stop":
                logger.info("Exotel AgentStream stop event received")
                end_reason = "COMPLETED" if call_done.is_set() else "DROPPED"
                break
            elif event == "dtmf":
                # DTMF is currently informational; voice is the supported input path.
                continue
    except WebSocketDisconnect:
        logger.info("Exotel AgentStream WebSocket disconnected")
        pass
    except ExotelProtocolError:
        end_reason = "DROPPED"
        logger.warning("Invalid Exotel AgentStream message; closing call WebSocket")
        await ws.close(code=4400, reason="Invalid audio message")
    except Exception:
        logger.exception("Exotel AgentStream session failed")
    finally:
        if audio_queue.full():
            try:
                audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        audio_queue.put_nowait(None)
        if prisma_task is not None and not prisma_task.done():
            prisma_task.cancel()
            await asyncio.gather(prisma_task, return_exceptions=True)
        if greeting_task is not None and not greeting_task.done():
            greeting_task.cancel()
            await asyncio.gather(greeting_task, return_exceptions=True)
        if session is not None:
            active_sessions.pop(session.session_id, None)
            await session.end(reason=end_reason)
            if end_reason == "COMPLETED":
                try:
                    await ws.close(code=1000)
                except Exception:
                    pass
        logger.info(
            "Exotel session ended (reason=%s, inbound_media_packets=%d)",
            end_reason,
            inbound_media_messages,
        )

