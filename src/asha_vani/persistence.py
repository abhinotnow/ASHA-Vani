"""Minimal Supabase persistence behind the existing async EventSink boundary."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from asha_vani.voice.events import CallEvent, CallEventType

logger = logging.getLogger(__name__)


class SupabaseEventSink:
    """Persist call sessions, per-call visit reports, and ordered events.

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
            visit_number = int(payload.get("visit_number", 1))
            patient = payload.get("patient", {})
            visit_values = payload.get("visit_values", {})
            triage = payload.get("triage", {})
            verification = payload.get("verification", "UNVERIFIED")
            patient_id = payload.get("matched_patient_id")
            if verification == "VERIFIED" and not patient_id and patient.get("full_name"):
                patient_id = self._upsert_patient(event, visit_number, patient)
            report = {
                "session_id": event.session_id,
                "visit_number": visit_number,
                "patient_id": patient_id,
                "patient": patient,
                "visit_values": visit_values,
                "triage": triage,
                "validation_issues": payload.get("validation_issues", []),
                "verification": verification,
                "recorded_at": event.timestamp,
                "is_simulated": event.is_simulated,
            }
            self._client.table("visits").upsert(
                {
                    "session_id": event.session_id,
                    "visit_number": visit_number,
                    "patient_id": patient_id,
                    "last_utterance_seq": payload.get("utterance_seq"),
                    "patient": patient,
                    "extracted_turn": payload.get("extracted_turn", {}),
                    "visit_values": visit_values,
                    "validation_issues": payload.get("validation_issues", []),
                    "triage": triage,
                    "verification": verification,
                    "next_action": payload.get("next_action"),
                    "final_report": report if verification == "VERIFIED" else {},
                    "finalized_at": event.timestamp if verification == "VERIFIED" else None,
                    "updated_at": event.timestamp,
                },
                on_conflict="session_id,visit_number",
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
                ).eq("session_id", event.session_id).eq(
                    "visit_number", int(payload.get("visit_number", 1))
                ).execute()

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

    def _upsert_patient(self, event: CallEvent, visit_number: int, patient: dict[str, Any]) -> str | None:
        """Create a distinct profile for a new identity; source key makes retries safe."""
        source_key = f"{event.session_id}:{visit_number}"
        result = self._client.table("patients").upsert(
            {
                "source_visit_key": source_key,
                "full_name": patient.get("full_name"),
                "spouse_or_guardian_name": patient.get("spouse_or_guardian_name"),
                "age_years": patient.get("age_years"),
                "village": patient.get("village"),
                "tola": patient.get("tola"),
            },
            on_conflict="source_visit_key",
        ).execute()
        rows = getattr(result, "data", None) or []
        if rows and rows[0].get("id"):
            return str(rows[0]["id"])
        lookup = self._client.table("patients").select("id").eq("source_visit_key", source_key).execute()
        rows = getattr(lookup, "data", None) or []
        return str(rows[0]["id"]) if rows else None

    async def load_register(self) -> dict[str, list[dict[str, Any]]]:
        """Fetch saved visit snapshots and their event history for console hydration."""
        return await asyncio.to_thread(self._load_register)

    def _load_register(self) -> dict[str, list[dict[str, Any]]]:
        visits_result = self._client.table("visits").select("*").order("updated_at", desc=True).execute()
        events_result = self._client.table("call_events").select("*").order("occurred_at").execute()
        calls_result = self._client.table("calls").select("*").order("started_at", desc=True).execute()
        return {
            "visits": list(visits_result.data or []),
            "events": list(events_result.data or []),
            "calls": list(calls_result.data or []),
        }

    async def find_patient_history(self, patient: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
        """Return exact identity candidates; callers must ask the ASHA to confirm."""
        return await asyncio.to_thread(self._find_patient_history, patient, limit)

    def _find_patient_history(self, patient: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        name = _normalize_identity(patient.get("full_name"))
        village = _normalize_identity(patient.get("village"))
        guardian = _normalize_identity(patient.get("spouse_or_guardian_name"))
        if not name or not village:
            return []
        result = (
            self._client.table("visits")
            .select("*")
            .eq("verification", "VERIFIED")
            .order("updated_at", desc=True)
            .limit(1000)
            .execute()
        )
        matches_by_patient: dict[str, dict[str, Any]] = {}
        for row in result.data or []:
            prior = row.get("patient") or {}
            if _normalize_identity(prior.get("full_name")) != name:
                continue
            if _normalize_identity(prior.get("village")) != village:
                continue
            prior_guardian = _normalize_identity(prior.get("spouse_or_guardian_name"))
            if guardian and prior_guardian and guardian != prior_guardian:
                continue
            patient_id = str(row.get("patient_id") or "")
            if not patient_id:
                continue
            row["final_report"] = row.get("final_report") or {
                "patient": prior,
                "visit_values": row.get("visit_values", {}),
                "triage": row.get("triage", {}),
                "verification": row.get("verification"),
            }
            matches_by_patient.setdefault(patient_id, row)
        matches = list(matches_by_patient.values())
        matches.sort(
            key=lambda row: (
                _normalize_identity((row.get("patient") or {}).get("spouse_or_guardian_name")) == guardian and bool(guardian),
                str(row.get("updated_at", "")),
            ),
            reverse=True,
        )
        return matches[: max(1, min(limit, 10))]


def _normalize_identity(value: Any) -> str:
    return re.sub(r"[^\w]+", "", str(value or "").casefold(), flags=re.UNICODE)
