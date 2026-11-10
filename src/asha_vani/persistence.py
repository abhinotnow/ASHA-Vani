"""Minimal Supabase persistence behind the existing async EventSink boundary."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import re
import time
from typing import Any

from asha_vani.extraction import transliterate_to_english
from asha_vani.voice.events import CallEvent, CallEventType

logger = logging.getLogger(__name__)
_LOCAL_CACHE_PATH = Path(__file__).resolve().parents[2] / ".local_register_cache.json"
_OFFLINE_COOLDOWN_S = 120.0


class SupabaseEventSink:
    """Persist call sessions, per-call visit reports, and ordered events.

    The Supabase Python client is synchronous, so each DB operation runs in a
    worker thread. Failed events are kept in a bounded in-memory queue and
    retried before the next event; a database outage does not break the call.
    """

    def __init__(self, url: str, service_role_key: str, *, client: Any | None = None) -> None:
        self._url = url
        self._service_role_key = service_role_key
        self._custom_client = client is not None
        if client is None:
            from supabase import create_client

            client = create_client(url, service_role_key)
        self._client = client
        self._pending: list[CallEvent] = []
        self._pending_limit = 500
        self._lock = asyncio.Lock()
        self._warned_unreachable = False
        self._offline_until: float = 0.0
        self._cached_register: dict[str, list[dict[str, Any]]] = self._load_disk_cache()

    def _load_disk_cache(self) -> dict[str, list[dict[str, Any]]]:
        if not self._custom_client and _LOCAL_CACHE_PATH.exists():
            try:
                raw = json.loads(_LOCAL_CACHE_PATH.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return _sanitize_register_payload(
                        {
                            "visits": list(raw.get("visits") or []),
                            "events": list(raw.get("events") or []),
                            "calls": list(raw.get("calls") or []),
                        }
                    )
            except Exception:
                pass
        return {"visits": [], "events": [], "calls": []}

    def _save_disk_cache(self) -> None:
        if self._custom_client:
            return
        try:
            _LOCAL_CACHE_PATH.write_text(
                json.dumps(self._cached_register, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _reconnect_client_if_possible(self) -> None:
        if self._custom_client:
            return
        try:
            from supabase import create_client

            self._client = create_client(self._url, self._service_role_key)
        except Exception:
            pass

    async def emit(self, event: CallEvent) -> None:
        async with self._lock:
            self._record_in_local_cache(event)
            if not self._custom_client and time.monotonic() < self._offline_until:
                self._pending.append(event)
                self._pending = self._pending[-self._pending_limit :]
                return
            batch = [*self._pending, event]
            self._pending = []
            for index, item in enumerate(batch):
                try:
                    await asyncio.to_thread(self._persist_one, item)
                    self._warned_unreachable = False
                except Exception as exc:
                    if not self._warned_unreachable:
                        logger.info(
                            "Supabase cloud unreachable (%s); using local cache (.local_register_cache.json)",
                            exc.__class__.__name__,
                        )
                        self._warned_unreachable = True
                    self._offline_until = time.monotonic() + _OFFLINE_COOLDOWN_S
                    self._reconnect_client_if_possible()
                    self._pending.extend(batch[index:])
                    self._pending = self._pending[-self._pending_limit :]
                    break

    def _record_in_local_cache(self, event: CallEvent) -> None:
        payload = event.payload
        self._cached_register["events"].append(
            {
                "session_id": event.session_id,
                "seq": event.seq,
                "event_type": event.event_type.value,
                "occurred_at": event.timestamp,
                "is_simulated": event.is_simulated,
                "payload": payload,
            }
        )
        if event.event_type is CallEventType.CALL_STARTED:
            self._cached_register["calls"].insert(
                0,
                {
                    "session_id": event.session_id,
                    "caller": payload.get("caller"),
                    "source": payload.get("source", "BROWSER_SIM"),
                    "is_simulated": event.is_simulated,
                    "started_at": event.timestamp,
                },
            )
        elif event.event_type is CallEventType.TURN_COMPLETED:
            visit_number = int(payload.get("visit_number", 1))
            row = {
                "session_id": event.session_id,
                "visit_number": visit_number,
                "patient_id": payload.get("matched_patient_id"),
                "patient": payload.get("patient", {}),
                "visit_values": payload.get("visit_values", {}),
                "triage": payload.get("triage", {}),
                "verification": payload.get("verification", "UNVERIFIED"),
                "updated_at": event.timestamp,
            }
            existing = next(
                (
                    v
                    for v in self._cached_register["visits"]
                    if v.get("session_id") == event.session_id and v.get("visit_number") == visit_number
                ),
                None,
            )
            if existing is not None:
                existing.update(row)
            else:
                self._cached_register["visits"].insert(0, row)
        self._save_disk_cache()

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
        async with self._lock:
            if not self._custom_client and time.monotonic() < self._offline_until:
                return dict(self._cached_register)
            try:
                data = await asyncio.to_thread(self._load_register)
                self._cached_register = data
                self._warned_unreachable = False
                self._save_disk_cache()
                return data
            except Exception as exc:
                if not self._warned_unreachable:
                    logger.info(
                        "Supabase cloud endpoint unreachable (%s); using local register cache",
                        exc.__class__.__name__,
                    )
                    self._warned_unreachable = True
                self._offline_until = time.monotonic() + _OFFLINE_COOLDOWN_S
                self._reconnect_client_if_possible()
                return dict(self._cached_register)

    def _load_register(self) -> dict[str, list[dict[str, Any]]]:
        visits_result = self._client.table("visits").select("*").order("updated_at", desc=True).execute()
        events_result = self._client.table("call_events").select("*").order("occurred_at").execute()
        calls_result = self._client.table("calls").select("*").order("started_at", desc=True).execute()
        return _sanitize_register_payload(
            {
                "visits": list(visits_result.data or []),
                "events": list(events_result.data or []),
                "calls": list(calls_result.data or []),
            }
        )

    async def find_patient_history(self, patient: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
        """Return exact identity candidates; callers must ask the ASHA to confirm."""
        async with self._lock:
            if not self._custom_client and time.monotonic() < self._offline_until:
                return []
            try:
                return await asyncio.to_thread(self._find_patient_history, patient, limit)
            except Exception:
                self._offline_until = time.monotonic() + _OFFLINE_COOLDOWN_S
                self._reconnect_client_if_possible()
                return []

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


def _english_patient_dict(patient: Any) -> dict[str, Any]:
    if not isinstance(patient, dict):
        return {}
    out = dict(patient)
    for key in ("full_name", "spouse_or_guardian_name", "village", "tola"):
        val = out.get(key)
        if isinstance(val, str) and val.strip():
            roman = transliterate_to_english(val).strip()
            if roman:
                out[key] = roman.title()
    return out


def _sanitize_register_payload(data: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    for visit in data.get("visits", []):
        if isinstance(visit, dict):
            if isinstance(visit.get("patient"), dict):
                visit["patient"] = _english_patient_dict(visit["patient"])
            if isinstance(visit.get("final_report"), dict) and isinstance(visit["final_report"].get("patient"), dict):
                visit["final_report"]["patient"] = _english_patient_dict(visit["final_report"]["patient"])
    for ev in data.get("events", []):
        if isinstance(ev, dict) and isinstance(ev.get("payload"), dict):
            payload = ev["payload"]
            if isinstance(payload.get("patient"), dict):
                payload["patient"] = _english_patient_dict(payload["patient"])
            if isinstance(payload.get("final_report"), dict) and isinstance(payload["final_report"].get("patient"), dict):
                payload["final_report"]["patient"] = _english_patient_dict(payload["final_report"]["patient"])
    return data
