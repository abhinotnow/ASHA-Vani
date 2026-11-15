"""ASHA worker registry with phone-to-worker lookup, E.164 normalization, and offline JSON fallback."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time
from typing import Any
import uuid

from pydantic import Field

from asha_vani.schemas import ProjectModel

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SEED_PATH = _REPO_ROOT / "data" / "asha_workers.json"
_DEFAULT_LOCAL_PATH = _REPO_ROOT / "data" / "asha_workers.local.json"
_OFFLINE_COOLDOWN_S = 120.0
_CACHE_TTL_S = 300.0


class PhoneEntry(ProjectModel):
    phone: str
    label: str = "own"
    verified: bool = True


class AshaWorker(ProjectModel):
    asha_id: str
    full_name: str
    language: str = "hi"
    state: str | None = None
    district: str | None = None
    phc: str | None = None
    active: bool = True
    is_simulated: bool = True
    phones: list[PhoneEntry] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        return f"{self.full_name} ({self.asha_id})"

    @property
    def short_name(self) -> str:
        return self.full_name.strip()


def normalize_phone(raw: str | None) -> str | None:
    """Normalize Indian mobile numbers to `+91XXXXXXXXXX`, or return None if invalid/withheld."""
    if not raw:
        return None
    digits = re.sub(r"[^\d]", "", str(raw).strip())
    if len(digits) == 10 and digits[0] in "6789":
        return f"+91{digits}"
    if len(digits) == 11 and digits.startswith("0") and digits[1] in "6789":
        return f"+91{digits[1:]}"
    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return f"+{digits}"
    return None


def mask_phone(raw: str | None) -> str:
    """Mask phone numbers for UI & logs (e.g. `+91 98•••••3210`)."""
    e164 = normalize_phone(raw)
    if not e164:
        return "UNKNOWN"
    local = e164[3:]
    return f"+91 {local[:2]}•••••{local[-4:]}"


def normalize_asha_id(raw: str | None) -> str | None:
    """Normalize `'005'` or `'ASHA-5'` into `'ASHA-005'` (supports 1-4 digits)."""
    if not raw:
        return None
    m = re.search(r"(\d{1,4})", str(raw).strip())
    if not m:
        return None
    num = int(m.group(1))
    return f"ASHA-{num:03d}" if num < 1000 else f"ASHA-{num}"


class WorkerRegistry:
    """Read-through registry backed by Supabase (`asha_workers` + `asha_worker_phones`) and JSON fallback."""

    def __init__(
        self,
        *,
        seed_path: Path = _DEFAULT_SEED_PATH,
        local_path: Path = _DEFAULT_LOCAL_PATH,
        supabase_client: Any | None = None,
    ) -> None:
        self._seed_path = seed_path
        self._local_path = local_path
        self._client = supabase_client
        self._offline_until: float = 0.0
        self._cache_loaded_at: float = 0.0
        self._workers_by_id: dict[str, AshaWorker] = {}
        self._workers_by_phone: dict[str, list[AshaWorker]] = {}
        self._pending_links: list[dict[str, Any]] = []
        self._load_json_fallback()

    def _load_json_fallback(self) -> None:
        merged: dict[str, dict[str, Any]] = {}
        for path, default_sim in ((self._seed_path, True), (self._local_path, False)):
            if not path.exists():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                for item in raw.get("workers", []):
                    aid = normalize_asha_id(item.get("asha_id")) or str(item.get("asha_id", "")).strip()
                    if not aid:
                        continue
                    entry = dict(item)
                    entry["asha_id"] = aid
                    entry.setdefault("is_simulated", default_sim)
                    merged[aid] = entry
            except Exception as exc:
                logger.warning("Failed loading worker seed file %s: %s", path, exc)

        workers: dict[str, AshaWorker] = {}
        by_phone: dict[str, list[AshaWorker]] = {}
        for aid, raw_worker in merged.items():
            phones: list[PhoneEntry] = []
            for p in raw_worker.get("phones", []):
                if isinstance(p, str):
                    norm = normalize_phone(p)
                    if norm:
                        phones.append(PhoneEntry(phone=norm))
                elif isinstance(p, dict):
                    norm = normalize_phone(p.get("phone") or p.get("phone_e164"))
                    if norm:
                        phones.append(
                            PhoneEntry(
                                phone=norm,
                                label=str(p.get("label", "own")),
                                verified=bool(p.get("verified", True)),
                            )
                        )
            worker = AshaWorker(
                asha_id=aid,
                full_name=str(raw_worker.get("full_name", aid)),
                language=str(raw_worker.get("language", "hi")),
                state=raw_worker.get("state"),
                district=raw_worker.get("district"),
                phc=raw_worker.get("phc"),
                active=bool(raw_worker.get("active", True)),
                is_simulated=bool(raw_worker.get("is_simulated", True)),
                phones=phones,
            )
            if not worker.active:
                continue
            workers[aid] = worker
            for pe in phones:
                if pe.verified:
                    by_phone.setdefault(pe.phone, []).append(worker)

        self._workers_by_id = workers
        self._workers_by_phone = by_phone

    def _refresh_from_supabase_if_needed(self, force: bool = False) -> None:
        if self._client is None:
            return
        now = time.monotonic()
        if not force and now < self._offline_until:
            return
        if not force and self._cache_loaded_at > 0 and (now - self._cache_loaded_at) < _CACHE_TTL_S:
            return
        try:
            w_res = self._client.table("asha_workers").select("*").eq("active", True).execute()
            p_res = self._client.table("asha_worker_phones").select("*").eq("verified", True).execute()
            w_rows = getattr(w_res, "data", None) or []
            p_rows = getattr(p_res, "data", None) or []
            if not w_rows:
                # Table exists but hasn't been seeded yet; keep JSON fallback
                self._cache_loaded_at = now
                return

            phones_by_id: dict[str, list[PhoneEntry]] = {}
            for pr in p_rows:
                norm = normalize_phone(pr.get("phone_e164"))
                aid = str(pr.get("asha_id") or "")
                if norm and aid:
                    phones_by_id.setdefault(aid, []).append(
                        PhoneEntry(phone=norm, label=str(pr.get("label", "own")), verified=True)
                    )

            workers: dict[str, AshaWorker] = {}
            by_phone: dict[str, list[AshaWorker]] = {}
            for wr in w_rows:
                aid = str(wr["asha_id"])
                phones = phones_by_id.get(aid, [])
                worker = AshaWorker(
                    asha_id=aid,
                    full_name=str(wr["full_name"]),
                    language=str(wr.get("language", "hi")),
                    state=wr.get("state"),
                    district=wr.get("district"),
                    phc=wr.get("phc"),
                    active=bool(wr.get("active", True)),
                    is_simulated=bool(wr.get("is_simulated", True)),
                    phones=phones,
                )
                workers[aid] = worker
                for pe in phones:
                    by_phone.setdefault(pe.phone, []).append(worker)

            self._workers_by_id = workers
            self._workers_by_phone = by_phone
            self._cache_loaded_at = now
        except Exception as exc:
            logger.info("WorkerRegistry Supabase unavailable (%s); using JSON fallback", exc.__class__.__name__)
            self._offline_until = now + _OFFLINE_COOLDOWN_S
            self._load_json_fallback()

    def lookup_phone(self, raw_phone: str | None) -> list[AshaWorker]:
        """Return 0, 1, or many active workers registered to `raw_phone`."""
        norm = normalize_phone(raw_phone)
        if not norm:
            return []
        self._refresh_from_supabase_if_needed()
        return list(self._workers_by_phone.get(norm, []))

    def by_asha_id(self, raw_id: str | None) -> AshaWorker | None:
        aid = normalize_asha_id(raw_id)
        if not aid:
            return None
        self._refresh_from_supabase_if_needed()
        return self._workers_by_id.get(aid)

    def list_workers(self) -> list[AshaWorker]:
        self._refresh_from_supabase_if_needed()
        return sorted(self._workers_by_id.values(), key=lambda w: w.asha_id)

    def queue_phone_link(self, raw_phone: str | None, asha_id: str, session_id: str | None = None) -> dict[str, Any] | None:
        """Queue an unverified phone->worker claim for PHC staff approval (never auto-links)."""
        norm = normalize_phone(raw_phone)
        aid = normalize_asha_id(asha_id)
        if not norm or not aid:
            return None
        if aid not in self._workers_by_id:
            return None
        # Do not queue if already linked
        if any(w.asha_id == aid for w in self.lookup_phone(norm)):
            return None

        record = {
            "id": str(uuid.uuid4()),
            "phone_e164": norm,
            "phone_masked": mask_phone(norm),
            "claimed_asha_id": aid,
            "worker_name": self._workers_by_id[aid].full_name,
            "session_id": session_id,
            "status": "PENDING",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if self._client is not None and time.monotonic() >= self._offline_until:
            try:
                res = (
                    self._client.table("pending_phone_links")
                    .upsert(
                        {
                            "phone_e164": norm,
                            "claimed_asha_id": aid,
                            "session_id": session_id,
                            "status": "PENDING",
                        },
                        on_conflict="phone_e164,claimed_asha_id,status",
                    )
                    .execute()
                )
                rows = getattr(res, "data", None) or []
                if rows:
                    record["id"] = str(rows[0].get("id", record["id"]))
            except Exception as exc:
                logger.info("Could not persist pending_phone_links (%s); keeping in-memory", exc.__class__.__name__)
                self._offline_until = time.monotonic() + _OFFLINE_COOLDOWN_S

        if not any(
            p["phone_e164"] == norm and p["claimed_asha_id"] == aid and p["status"] == "PENDING"
            for p in self._pending_links
        ):
            self._pending_links.insert(0, record)
        return record

    def list_pending_links(self, status: str = "PENDING") -> list[dict[str, Any]]:
        if self._client is not None and time.monotonic() >= self._offline_until:
            try:
                res = (
                    self._client.table("pending_phone_links")
                    .select("*")
                    .eq("status", status)
                    .order("created_at", desc=True)
                    .execute()
                )
                rows = getattr(res, "data", None) or []
                out: list[dict[str, Any]] = []
                for r in rows:
                    aid = str(r.get("claimed_asha_id", ""))
                    worker = self._workers_by_id.get(aid)
                    out.append(
                        {
                            "id": str(r["id"]),
                            "phone_masked": mask_phone(r.get("phone_e164")),
                            "claimed_asha_id": aid,
                            "worker_name": worker.full_name if worker else aid,
                            "session_id": r.get("session_id"),
                            "status": r.get("status", status),
                            "created_at": r.get("created_at"),
                        }
                    )
                return out
            except Exception:
                self._offline_until = time.monotonic() + _OFFLINE_COOLDOWN_S
        return [
            {k: v for k, v in p.items() if k != "phone_e164"}
            for p in self._pending_links
            if p.get("status") == status
        ]

    def approve_link(self, link_id: str) -> bool:
        return self._resolve_link(link_id, approve=True)

    def reject_link(self, link_id: str) -> bool:
        return self._resolve_link(link_id, approve=False)

    def _resolve_link(self, link_id: str, *, approve: bool) -> bool:
        new_status = "APPROVED" if approve else "REJECTED"
        now_iso = datetime.now(timezone.utc).isoformat()
        target_phone: str | None = None
        target_aid: str | None = None

        for p in self._pending_links:
            if p.get("id") == link_id and p.get("status") == "PENDING":
                p["status"] = new_status
                p["resolved_at"] = now_iso
                target_phone = p.get("phone_e164")
                target_aid = p.get("claimed_asha_id")
                break

        if self._client is not None and time.monotonic() >= self._offline_until:
            try:
                res = (
                    self._client.table("pending_phone_links")
                    .update({"status": new_status, "resolved_at": now_iso})
                    .eq("id", link_id)
                    .execute()
                )
                rows = getattr(res, "data", None) or []
                if rows:
                    target_phone = rows[0].get("phone_e164", target_phone)
                    target_aid = rows[0].get("claimed_asha_id", target_aid)
                if approve and target_phone and target_aid:
                    self._client.table("asha_worker_phones").upsert(
                        {
                            "phone_e164": target_phone,
                            "asha_id": target_aid,
                            "label": "own",
                            "verified": True,
                        },
                        on_conflict="phone_e164,asha_id",
                    ).execute()
                    self._refresh_from_supabase_if_needed(force=True)
                    return True
            except Exception:
                self._offline_until = time.monotonic() + _OFFLINE_COOLDOWN_S

        if approve and target_phone and target_aid:
            worker = self._workers_by_id.get(target_aid)
            if worker is not None:
                if not any(pe.phone == target_phone for pe in worker.phones):
                    worker.phones.append(PhoneEntry(phone=target_phone, label="own", verified=True))
                existing = self._workers_by_phone.setdefault(target_phone, [])
                if all(w.asha_id != target_aid for w in existing):
                    existing.append(worker)
            return True
        return target_aid is not None

