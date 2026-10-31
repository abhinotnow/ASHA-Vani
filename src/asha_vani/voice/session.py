"""CallSession orchestrator: ordered turn queue, strict half-duplex gate, and EventSink emissions.

Enforces AGENTS.md 4.2 Edge-Case Invariants:
1. Strict Half-Duplex Anti-Echo: drops incoming mic audio frames while TTS is playing
   PLUS a configurable tail mute window (default 200 ms) after TTS finishes.
2. Strict Turn Ordering: one sequential `turn_queue` worker per call; utterances never race.
3. Exact 1024-byte Framing: incoming audio bytes pass through `FrameBuffer`.
4. Turn-level Dashboard Updates: emits `call_started`, `turn_completed`, and `call_ended`
   to all registered `EventSink`s (no flickering partial transcripts).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import time
from typing import Any

from asha_vani.voice.audio import FRAME_BYTES, FrameBuffer
from asha_vani.voice.events import CallEvent, CallEventType, EventSink
from asha_vani.voice.stt import EndOfUtteranceDetector, STTChunk
from asha_vani.voice.tts import TextToSpeech
from asha_vani.voice.turn_runner import LangGraphTurnRunner, TurnOutput, TurnRunner, VerificationStatus

DEFAULT_TAIL_MUTE_S: float = 0.20  # 200 ms post-TTS tail mute


class HalfDuplexGate:
    """Controls microphone muting during TTS playback + a 200 ms anti-echo tail window."""

    def __init__(
        self,
        tail_mute_s: float = DEFAULT_TAIL_MUTE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.tail_mute_s = tail_mute_s
        self._clock = clock
        self._speaking: bool = False
        self._mute_until: float = 0.0

    def start_speaking(self) -> None:
        self._speaking = True

    def stop_speaking(self) -> None:
        self._speaking = False
        self._mute_until = self._clock() + self.tail_mute_s

    @property
    def is_mic_muted(self) -> bool:
        if self._speaking:
            return True
        return self._clock() < self._mute_until


class CallSession:
    """Manages one live or replayed ASHA phone call session."""

    def __init__(
        self,
        session_id: str,
        *,
        caller: str = "+919900000001",
        source: str = "BROWSER_SIM",
        turn_runner: TurnRunner | None = None,
        tts: TextToSpeech | None = None,
        sinks: list[EventSink] | None = None,
        tail_mute_s: float = DEFAULT_TAIL_MUTE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session_id = session_id
        self.caller = caller
        self.source = source
        self.turn_runner: TurnRunner = turn_runner or LangGraphTurnRunner()
        self.tts = tts
        self.sinks: list[EventSink] = list(sinks or [])
        self._clock = clock

        self.duplex_gate = HalfDuplexGate(tail_mute_s=tail_mute_s, clock=clock)
        self.frame_buffer = FrameBuffer(frame_bytes=FRAME_BYTES)
        self.eou_detector = EndOfUtteranceDetector()

        self.accepted_frames: list[bytes] = []
        self.dropped_mic_bytes: int = 0
        self.speaker_out_frames: list[bytes] = []
        self.turn_history: list[TurnOutput] = []

        self._event_seq: int = 0
        self._utterance_seq: int = 0
        self._started_at: float = self._clock()
        self._turn_lock = asyncio.Lock()

    async def _emit(self, event_type: CallEventType, payload: dict[str, Any]) -> CallEvent:
        self._event_seq += 1
        event = CallEvent(
            session_id=self.session_id,
            seq=self._event_seq,
            event_type=event_type,
            is_simulated=True,
            payload=payload,
        )
        for sink in self.sinks:
            await sink.emit(event)
        return event

    async def start(self) -> CallEvent:
        """Emit `call_started` when the call connects."""
        self._started_at = self._clock()
        return await self._emit(
            CallEventType.CALL_STARTED,
            {"caller": self.caller, "source": self.source},
        )

    def push_mic_audio(self, pcm16_bytes: bytes) -> list[bytes]:
        """Receive incoming mic bytes. Drops frames if half-duplex gate is muted; else yields 1024-byte frames."""
        if self.duplex_gate.is_mic_muted:
            self.dropped_mic_bytes += len(pcm16_bytes)
            self.frame_buffer.clear()
            self.eou_detector.clear()
            return []

        frames = self.frame_buffer.push(pcm16_bytes)
        self.accepted_frames.extend(frames)
        return frames

    async def receive_stt_chunk(
        self,
        chunk: STTChunk | str,
        *,
        stt_latency_ms: float = 0.0,
    ) -> TurnOutput | None:
        """Feed an STT chunk into the EOU detector; if an utterance completes, run the turn sequentially."""
        if self.duplex_gate.is_mic_muted:
            return None

        stt_chunk = chunk if isinstance(chunk, STTChunk) else STTChunk(text=chunk, is_final=True)
        completed_text = self.eou_detector.feed(stt_chunk)
        if completed_text is None:
            return None

        return await self.process_utterance(completed_text, stt_latency_ms=stt_latency_ms)

    async def process_utterance(
        self,
        transcript: str,
        *,
        stt_latency_ms: float = 0.0,
    ) -> TurnOutput | None:
        """Run one completed utterance through TurnRunner + TTS in strict sequential order."""
        cleaned, _ = self.eou_detector.feed(STTChunk(text=transcript, is_final=True)) or transcript, False
        text = (cleaned or transcript).strip()
        if not text:
            return None

        async with self._turn_lock:
            self._utterance_seq += 1
            seq = self._utterance_seq

            t0 = self._clock()
            turn_out = await self.turn_runner.run_turn(
                session_id=self.session_id,
                utterance_seq=seq,
                transcript=text,
            )
            graph_latency_ms = round((self._clock() - t0) * 1000.0, 2)
            self.turn_history.append(turn_out)

            # Synthesize TTS reply under strict half-duplex speaker gate
            tts_first_byte_ms = 0.0
            if self.tts is not None and turn_out.reply_text:
                self.duplex_gate.start_speaking()
                self.frame_buffer.clear()
                self.eou_detector.clear()
                tts_t0 = self._clock()
                first_chunk = True
                try:
                    async for audio_frame in self.tts.synthesize(turn_out.reply_text):
                        if first_chunk:
                            tts_first_byte_ms = round((self._clock() - tts_t0) * 1000.0, 2)
                            first_chunk = False
                        self.speaker_out_frames.append(audio_frame)
                finally:
                    self.duplex_gate.stop_speaking()

            await self._emit(
                CallEventType.TURN_COMPLETED,
                {
                    "utterance_seq": seq,
                    "stt_transcript": turn_out.transcript,
                    "patient": turn_out.patient.model_dump(),
                    "extracted_turn": turn_out.extracted_turn.model_dump(),
                    "visit_values": turn_out.visit_values.model_dump(),
                    "validation_issues": [
                        issue.model_dump() for issue in turn_out.validation_result.issues
                    ],
                    "triage": turn_out.triage_result.model_dump(),
                    "next_action": turn_out.next_action,
                    "verification": turn_out.verification.value,
                    "reply_text": turn_out.reply_text,
                    "latencies_ms": {
                        "stt": round(stt_latency_ms, 2),
                        "graph": graph_latency_ms,
                        "tts_first_byte": tts_first_byte_ms,
                    },
                },
            )
            return turn_out

    async def end(self, reason: str = "COMPLETED") -> CallEvent:
        """Close the call session. Marks unconfirmed visits as INCOMPLETE if call dropped."""
        duration_s = round(max(0.0, self._clock() - self._started_at), 2)
        last_verification = (
            self.turn_history[-1].verification.value if self.turn_history else None
        )
        if reason == "DROPPED" and last_verification == VerificationStatus.UNVERIFIED.value:
            last_verification = VerificationStatus.INCOMPLETE.value

        return await self._emit(
            CallEventType.CALL_ENDED,
            {
                "reason": reason,
                "duration_s": duration_s,
                "turns_count": len(self.turn_history),
                "final_verification": last_verification,
            },
        )

