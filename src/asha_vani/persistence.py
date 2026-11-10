"""Minimal Supabase persistence behind the existing async EventSink boundary."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from asha_vani.voice.events import CallEvent, CallEventType

logger = logging.getLogger(__name__)


class SupabaseEventSink:
    """Persist calls, the latest visit snapshot, and ordered call events.

    The Supabase Python client is synchronous, so each DB operation runs in a
    worker thread. Failed events are kept in a bounded in-memory queue and
    retried before the next event; a database outage does not break the call.
    """

    def __init__(self, url: str, service_role_key: str, *, client: Any | None = None) -> None:
        if client is None:
            from supabase import create_client

            client = create_client(url, service_role_key)
        self._client = client
        self._pending: list[CallEvent] = []
        self._pending_limit = 500
        self._lock = asyncio.Lock()

    async def emit(self, event: CallEvent) -> None:
        async with self._lock:
            batch = [*self._pending, event]
            self._pending = []
            for index, item in enumerate(batch):
                try:
                    await asyncio.to_thread(self._persist_one, item)
                except Exception:
                    logger.exception(
                        "Could not persist call event %s/%s; retaining for retry",
                        item.session_id,
                        item.seq,
                    )
                    self._pending.extend(batch[index:])
                    self._pending = self._pending[-self._pending_limit :]
                    break

    def _persist_one(self, event: CallEvent) -> None:
        payload = event.payload
        if event.event_type is CallEventType.CALL_STARTED:
            self._client.table("calls").upsert(
                {
                    "session_id": event.session_id,
                    "caller": payload.get("caller"),
                    "source": payload.get("source", "BROWSER_SIM"),
                    "is_simulated": event.is_simulated,
                    "started_at": event.timestamp,
                },
                on_conflict="session_id",
            ).execute()
        elif event.event_type is CallEventType.TURN_COMPLETED:
            self._client.table("visits").upsert(
                {
                    "session_id": event.session_id,
                    "last_utterance_seq": payload.get("utterance_seq"),
                    "patient": payload.get("patient", {}),
                    "extracted_turn": payload.get("extracted_turn", {}),
                    "visit_values": payload.get("visit_values", {}),
                    "validation_issues": payload.get("validation_issues", []),
                    "triage": payload.get("triage", {}),
                    "verification": payload.get("verification", "UNVERIFIED"),
                    "next_action": payload.get("next_action"),
                    "updated_at": event.timestamp,
                },
                on_conflict="session_id",
            ).execute()
        elif event.event_type is CallEventType.CALL_ENDED:
            self._client.table("calls").update(
                {
                    "ended_at": event.timestamp,
                    "end_reason": payload.get("reason", "COMPLETED"),
                }
            ).eq("session_id", event.session_id).execute()
            if payload.get("final_verification") == "INCOMPLETE":
                self._client.table("visits").update(
                    {"verification": "INCOMPLETE", "updated_at": event.timestamp}
                ).eq("session_id", event.session_id).execute()

        self._client.table("call_events").upsert(
            {
                "session_id": event.session_id,
                "seq": event.seq,
                "event_type": event.event_type.value,
                "occurred_at": event.timestamp,
                "is_simulated": event.is_simulated,
                "payload": payload,
            },
            on_conflict="session_id,seq",
        ).execute()

    async def load_register(self) -> dict[str, list[dict[str, Any]]]:
        """Fetch saved visit snapshots and their event history for console hydration."""
        return await asyncio.to_thread(self._load_register)

    def _load_register(self) -> dict[str, list[dict[str, Any]]]:
        visits_result = self._client.table("visits").select("*").order("updated_at", desc=True).execute()
        events_result = self._client.table("call_events").select("*").order("occurred_at").execute()
        return {
            "visits": list(visits_result.data or []),
            "events": list(events_result.data or []),
        }
