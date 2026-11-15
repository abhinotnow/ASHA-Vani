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
import re
import struct
import time
from typing import Any

from asha_vani.language.locale import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES, t
from asha_vani.language.registry import WorkerRegistry, mask_phone
from asha_vani.language.routing import (
    BILINGUAL_AUTO_GREETING,
    CallLanguage,
    LanguageSource,
    detect_utterance_language,
    is_not_me_utterance,
    parse_switch_request,
    resolve_call_language,
)
from asha_vani.voice.audio import FRAME_BYTES, SAMPLE_RATE_HZ, EnergyVAD, FrameBuffer
from asha_vani.voice.events import CallEvent, CallEventType, EventSink
from asha_vani.voice.stt import EndOfUtteranceDetector, STTChunk
from asha_vani.voice.tts import TextToSpeech
from asha_vani.voice.turn_runner import LangGraphTurnRunner, TurnOutput, TurnRunner, VerificationStatus

DEFAULT_TAIL_MUTE_S: float = 0.20  # 200 ms post-TTS tail mute
PCM16_BYTES_PER_SECOND: float = float(SAMPLE_RATE_HZ * 2)  # 16,000 bytes/s @ 8 kHz mono PCM16


def build_pcm16_wav(pcm16_bytes: bytes, sample_rate: int = SAMPLE_RATE_HZ) -> bytes:
    """Wrap raw 16-bit mono PCM bytes in a standard 44-byte RIFF WAVE header."""
    data_size = len(pcm16_bytes)
    byte_rate = sample_rate * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format
        1,  # 1 channel (mono)
        sample_rate,
        byte_rate,
        2,  # block align
        16,  # bits per sample
        b"data",
        data_size,
    )
    return header + pcm16_bytes


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

    def stop_speaking(self, extra_playback_s: float = 0.0) -> None:
        self._speaking = False
        self._mute_until = self._clock() + max(0.0, extra_playback_s) + self.tail_mute_s

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
        language: CallLanguage | None = None,
        registry: WorkerRegistry | None = None,
        tail_mute_s: float = DEFAULT_TAIL_MUTE_S,
        realtime_playback_hold: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session_id = session_id
        self.caller = caller
        self.source = source
        self.registry = registry
        if language is not None:
            self.language = language
        elif registry is not None:
            self.language = resolve_call_language(caller, registry)
        else:
            self.language = CallLanguage.from_code(DEFAULT_LANGUAGE)

        self.turn_runner: TurnRunner = turn_runner or LangGraphTurnRunner(registry=registry)
        if self.language.worker is not None and hasattr(self.turn_runner, "seed_worker"):
            self.turn_runner.seed_worker(session_id, self.language.worker)

        self.tts = tts
        self.sinks: list[EventSink] = list(sinks or [])
        self.realtime_playback_hold = realtime_playback_hold
        self._clock = clock
        self.language_changed_event = asyncio.Event()
        self._awaiting_language_choice: bool = False

        self.duplex_gate = HalfDuplexGate(tail_mute_s=tail_mute_s, clock=clock)
        self.frame_buffer = FrameBuffer(frame_bytes=FRAME_BYTES)
        self.eou_detector = EndOfUtteranceDetector()
        self.vad = EnergyVAD()
        self._vad_speech_ended: bool = False

        self.accepted_frames: list[bytes] = []
        self.dropped_mic_bytes: int = 0
        self.speaker_out_frames: list[bytes] = []
        self.call_audio_pcm16: bytearray = bytearray()
        self.turn_history: list[TurnOutput] = []

        self._event_seq: int = 0
        self._utterance_seq: int = 0
        self._started_at: float = self._clock()
        self._turn_lock = asyncio.Lock()

    def record_speaker_audio(self, pcm16_bytes: bytes) -> None:
        """Append synthesized TTS audio to the session's continuous call recording."""
        if not pcm16_bytes:
            return
        if len(pcm16_bytes) >= 44 and pcm16_bytes[:4] == b"RIFF":
            pcm16_bytes = pcm16_bytes[44:]
        self.speaker_out_frames.append(pcm16_bytes)
        self.call_audio_pcm16.extend(pcm16_bytes)

    def build_call_wav(self) -> bytes:
        """Return a complete 8 kHz 16-bit mono WAV of the entire call timeline."""
        return build_pcm16_wav(bytes(self.call_audio_pcm16), sample_rate=SAMPLE_RATE_HZ)

    async def _emit(self, event_type: CallEventType, payload: dict[str, Any]) -> CallEvent:
        self._event_seq += 1
        event = CallEvent(
            session_id=self.session_id,
            seq=self._event_seq,
            event_type=event_type,
            is_simulated=self.source != "EXOTEL",
            payload=payload,
        )
        for sink in self.sinks:
            await sink.emit(event)
        return event

    def greeting_text(self) -> str:
        """Return the initial greeting in the resolved language (or bilingual if auto)."""
        if self.language.code == "auto" and not self.language.locked:
            return BILINGUAL_AUTO_GREETING
        lang = self.language.effective_code
        if self.language.worker is not None:
            return t(
                lang,
                "greeting_known_worker",
                short_name=self.language.worker.short_name,
                asha_id=self.language.worker.asha_id,
            )
        return t(lang, "greeting")

    async def set_language(self, new_lang: CallLanguage) -> None:
        """Update session language, notify STT stream listeners, and emit `language_changed`."""
        old_code = self.language.code
        self.language = new_lang
        if hasattr(self.tts, "set_language"):
            self.tts.set_language(new_lang.effective_code)
        self.language_changed_event.set()
        await self._emit(
            CallEventType.LANGUAGE_CHANGED,
            {
                "from_language": old_code,
                "to_language": new_lang.code,
                "language": new_lang.effective_code,
                "bcp47": new_lang.bcp47,
                "language_source": new_lang.source.value,
                "asha_id": new_lang.worker.asha_id if new_lang.worker else None,
            },
        )

    async def _speak_reply(self, reply_text: str) -> tuple[float, int]:
        tts_first_byte_ms = 0.0
        tts_bytes_sent = 0
        if self.tts is not None and reply_text:
            self.duplex_gate.start_speaking()
            self.frame_buffer.clear()
            self.eou_detector.clear()
            tts_t0 = self._clock()
            first_chunk = True
            try:
                async for audio_frame in self.tts.synthesize(reply_text):
                    if first_chunk:
                        tts_first_byte_ms = round((self._clock() - tts_t0) * 1000.0, 2)
                        first_chunk = False
                    self.record_speaker_audio(audio_frame)
                    tts_bytes_sent += len(audio_frame)
            finally:
                extra_hold_s = 0.0
                if self.realtime_playback_hold and tts_bytes_sent > 0:
                    elapsed_s = max(0.0, self._clock() - tts_t0)
                    playback_s = tts_bytes_sent / PCM16_BYTES_PER_SECOND
                    extra_hold_s = max(0.0, playback_s - elapsed_s)
                self.duplex_gate.stop_speaking(extra_playback_s=extra_hold_s)
        return tts_first_byte_ms, tts_bytes_sent

    async def start(self) -> CallEvent:
        """Emit `call_started` when the call connects."""
        self._started_at = self._clock()
        return await self._emit(
            CallEventType.CALL_STARTED,
            {
                "caller": self.caller,
                "caller_masked": mask_phone(self.caller),
                "source": self.source,
                "language": self.language.effective_code,
                "language_raw": self.language.code,
                "language_source": self.language.source.value,
                "asha_id": self.language.worker.asha_id if self.language.worker else None,
                "asha_name": self.language.worker.display_name if self.language.worker else None,
            },
        )

    def push_mic_audio(self, pcm16_bytes: bytes) -> list[bytes]:
        """Receive incoming mic bytes. Drops frames if half-duplex gate is muted; else yields 1024-byte frames."""
        if self.duplex_gate.is_mic_muted:
            self.dropped_mic_bytes += len(pcm16_bytes)
            self.frame_buffer.clear()
            self.eou_detector.clear()
            self.vad.reset()
            self._vad_speech_ended = False
            return []

        frames = self.frame_buffer.push(pcm16_bytes)
        self.accepted_frames.extend(frames)
        for frame in frames:
            self.call_audio_pcm16.extend(frame)
            vad_signal = self.vad.process_frame(frame)
            if vad_signal == "SPEECH_END":
                self._vad_speech_ended = True
        return frames

    async def flush_on_vad(self) -> TurnOutput | None:
        """Triggered when VAD confirms user has finished speaking; flushes buffered speech."""
        if self.duplex_gate.is_mic_muted or not self.eou_detector.has_pending_speech:
            return None
        completed_text = self.eou_detector.flush()
        if not completed_text:
            return None
        return await self.process_utterance(completed_text)

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

        # If user is still actively speaking (VAD energy high) and didn't say "over", don't cut them off!
        if self.vad.is_speaking and not stt_chunk.is_final:
            return None

        return await self.process_utterance(
            completed_text,
            stt_latency_ms=stt_latency_ms,
            detected_language_code=stt_chunk.detected_language_code,
        )

    def _repeat_prompt_for_action(self, lang: str, action: str | None) -> str:
        if action in {"ASK_PATIENT_NAME", "ASK_SPOUSE_NAME", "ASK_WEEKS", "ASK_BP", "ASK_HB", "ASK_SYMPTOMS"}:
            return t(lang, action)
        if action == "ASK_ASHA_ID":
            return t(lang, "ask_asha_id")
        if action == "CONFIRM_VALUES":
            return t(lang, "collect_confirm_fallback")
        if action == "ASK_ANOTHER_PATIENT":
            return t(lang, "continue_prompt")
        return t(lang, "ASK_PATIENT_NAME")

    async def _handle_control_utterance(self, text: str) -> str | None:
        """Handle keypad 9 language menu, spoken switch requests, or 'not me' (0). Returns reply text if handled."""
        cleaned = text.strip()
        lang = self.language.effective_code

        if self._awaiting_language_choice:
            choice_map = {"1": "hi", "2": "ta", "3": "te"}
            digit = re.sub(r"[^\d]", "", cleaned)
            chosen = choice_map.get(digit) or parse_switch_request(cleaned)
            self._awaiting_language_choice = False
            if chosen:
                await self.set_language(
                    CallLanguage.from_code(
                        chosen,
                        source=LanguageSource.SWITCHED,
                        worker=self.language.worker,
                        shared_workers=self.language.shared_workers,
                        locked=True,
                    )
                )
                last_action = self.turn_runner.get_last_action(self.session_id) if hasattr(self.turn_runner, "get_last_action") else None
                return f"{t(chosen, 'language_switched')} {self._repeat_prompt_for_action(chosen, last_action)}"

        if re.match(r"^\s*9#?\s*$", cleaned):
            self._awaiting_language_choice = True
            return t(lang, "language_menu")

        switch_target = parse_switch_request(cleaned)
        if switch_target and len(cleaned.split()) <= 5:
            await self.set_language(
                CallLanguage.from_code(
                    switch_target,
                    source=LanguageSource.SWITCHED,
                    worker=self.language.worker,
                    shared_workers=self.language.shared_workers,
                    locked=True,
                )
            )
            last_action = self.turn_runner.get_last_action(self.session_id) if hasattr(self.turn_runner, "get_last_action") else None
            return f"{t(switch_target, 'language_switched')} {self._repeat_prompt_for_action(switch_target, last_action)}"

        if self.language.worker is not None and (re.match(r"^\s*0#?\s*$", cleaned) or is_not_me_utterance(cleaned, lang)):
            self.language.worker = None
            if hasattr(self.turn_runner, "clear_worker"):
                self.turn_runner.clear_worker(self.session_id)
            return t(lang, "ask_asha_id")

        return None

    async def process_utterance(
        self,
        transcript: str,
        *,
        stt_latency_ms: float = 0.0,
        detected_language_code: str | None = None,
    ) -> TurnOutput | None:
        """Run one completed utterance through TurnRunner + TTS in strict sequential order."""
        cleaned, _ = self.eou_detector.feed(STTChunk(text=transcript, is_final=True)) or transcript, False
        text = (cleaned or transcript).strip()
        if not text:
            return None

        async with self._turn_lock:
            if not self.language.locked:
                detected = detect_utterance_language(text, detected_language_code=detected_language_code)
                await self.set_language(
                    CallLanguage.from_code(
                        detected,
                        source=LanguageSource.AUTO_DETECTED,
                        worker=self.language.worker,
                        shared_workers=self.language.shared_workers,
                        locked=True,
                    )
                )

            control_reply = await self._handle_control_utterance(text)
            if control_reply is not None:
                await self._speak_reply(control_reply)
                last = self.turn_history[-1] if self.turn_history else None
                if last is not None:
                    return last.model_copy(
                        update={
                            "reply_text": control_reply,
                            "language": self.language.effective_code,
                            "language_source": self.language.source.value,
                        }
                    )
                return None

            self._utterance_seq += 1
            seq = self._utterance_seq

            t0 = self._clock()
            lang_code = self.language.effective_code
            try:
                turn_out = await self.turn_runner.run_turn(
                    session_id=self.session_id,
                    utterance_seq=seq,
                    transcript=text,
                    language=lang_code,
                    language_source=self.language.source.value,
                )
            except TypeError:
                turn_out = await self.turn_runner.run_turn(
                    session_id=self.session_id,
                    utterance_seq=seq,
                    transcript=text,
                )

            # If an unknown caller states a registered ASHA ID, adopt her language & queue pending phone link
            if (
                turn_out.asha_id
                and self.registry is not None
                and self.language.source is not LanguageSource.REGISTERED
            ):
                worker = self.registry.by_asha_id(turn_out.asha_id)
                if worker is not None:
                    self.registry.queue_phone_link(self.caller, worker.asha_id, self.session_id)
                    if worker.language != lang_code and self.language.source is not LanguageSource.SWITCHED:
                        await self.set_language(
                            CallLanguage.from_code(
                                worker.language,
                                source=LanguageSource.REGISTERED,
                                worker=worker,
                                locked=True,
                            )
                        )
                        lang_code = self.language.effective_code
                        turn_out = turn_out.model_copy(
                            update={
                                "language": lang_code,
                                "language_source": LanguageSource.REGISTERED.value,
                                "reply_text": t(
                                    lang_code,
                                    "greet_worker_ask_patient",
                                    short_name=worker.short_name,
                                    asha_id=worker.asha_id,
                                ),
                            }
                        )

            graph_latency_ms = round((self._clock() - t0) * 1000.0, 2)
            self.turn_history.append(turn_out)

            tts_first_byte_ms, _ = await self._speak_reply(turn_out.reply_text)

            await self._emit(
                CallEventType.TURN_COMPLETED,
                {
                    "utterance_seq": seq,
                    "visit_number": turn_out.visit_number,
                    "matched_patient_id": turn_out.matched_patient_id,
                    "stt_transcript": turn_out.transcript,
                    "asha_id": turn_out.asha_id,
                    "asha_name": turn_out.asha_name,
                    "language": self.language.effective_code,
                    "language_source": self.language.source.value,
                    "visit_category": turn_out.visit_category,
                    "is_pregnant": turn_out.is_pregnant,
                    "english_summary": turn_out.english_summary,
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
        """Close the call session. Any active unconfirmed visit is marked INCOMPLETE."""
        duration_s = round(max(0.0, self._clock() - self._started_at), 2)
        last_verification = (
            self.turn_history[-1].verification.value if self.turn_history else None
        )
        if last_verification == VerificationStatus.UNVERIFIED.value:
            last_verification = VerificationStatus.INCOMPLETE.value

        return await self._emit(
            CallEventType.CALL_ENDED,
            {
                "reason": reason,
                "duration_s": duration_s,
                "turns_count": len(self.turn_history),
                "final_verification": last_verification,
                "visit_number": self.turn_history[-1].visit_number if self.turn_history else 1,
                "source": self.source,
            },
        )

