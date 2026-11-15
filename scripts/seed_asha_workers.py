"""Idempotent seed script for `asha_workers` and `asha_worker_phones` in Supabase."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from asha_vani.language.registry import normalize_asha_id, normalize_phone

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FILES = (
    (_REPO_ROOT / "data" / "asha_workers.json", True),
    (_REPO_ROOT / "data" / "asha_workers.local.json", False),
)


def main() -> None:
    load_dotenv()
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")

    from supabase import create_client

    client = create_client(url, key)
    workers_count = 0
    phones_count = 0

    for path, default_sim in _FILES:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("workers", []):
            aid = normalize_asha_id(item.get("asha_id")) or str(item.get("asha_id", "")).strip()
            if not aid:
                continue
            client.table("asha_workers").upsert(
                {
                    "asha_id": aid,
                    "full_name": item["full_name"],
                    "language": item.get("language", "hi"),
                    "state": item.get("state"),
                    "district": item.get("district"),
                    "phc": item.get("phc"),
                    "active": bool(item.get("active", True)),
                    "is_simulated": bool(item.get("is_simulated", default_sim)),
                },
                on_conflict="asha_id",
            ).execute()
            workers_count += 1
            for p in item.get("phones", []):
                raw = p if isinstance(p, str) else p.get("phone")
                norm = normalize_phone(raw)
                if not norm:
                    continue
                label = "own" if isinstance(p, str) else str(p.get("label", "own"))
                client.table("asha_worker_phones").upsert(
                    {
                        "phone_e164": norm,
                        "asha_id": aid,
                        "label": label,
                        "verified": True,
                    },
                    on_conflict="phone_e164,asha_id",
                ).execute()
                phones_count += 1

    print(f"Seeded {workers_count} workers and {phones_count} phone links.")


if __name__ == "__main__":
    main()

