"""Turn-level CallEvent megaphone (`EventSink`) for decoupling CallSession from DB & UI.

Design invariant (AGENTS.md 4.2):
- No flickering partial transcripts are sent to the dashboard or database.
- Only structured turn-level events (`call_started`, `turn_completed`, `call_ended`) are emitted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from pydantic import Field

from asha_vani.schemas import ProjectModel


class CallEventType(str, Enum):
    CALL_STARTED = "call_started"
    TURN_COMPLETED = "turn_completed"
    CALL_ENDED = "call_ended"


class CallEvent(ProjectModel):
    """Structured event emitted by CallSession to all registered EventSinks."""

    session_id: str
    seq: int
    event_type: CallEventType
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    is_simulated: bool = True
    payload: dict[str, Any] = Field(default_factory=dict)


class EventSink(Protocol):
    """Protocol for any listener (in-memory logger, WebSocket broadcaster, or Supabase DB writer)."""

    async def emit(self, event: CallEvent) -> None: ...


class InMemoryEventSink:
    """Stores emitted events in memory for unit tests, replay inspection, and latency benchmarking."""

    def __init__(self) -> None:
        self.events: list[CallEvent] = []

    async def emit(self, event: CallEvent) -> None:
        self.events.append(event)

    def by_type(self, event_type: CallEventType) -> list[CallEvent]:
        return [e for e in self.events if e.event_type is event_type]

