from __future__ import annotations

import asyncio
from types import SimpleNamespace

from asha_vani.persistence import SupabaseEventSink
from asha_vani.voice.events import CallEvent, CallEventType


class FakeSupabase:
    def __init__(self) -> None:
        self.rows: dict[str, list[dict]] = {"calls": [], "visits": [], "call_events": [], "patients": []}

    def table(self, name: str) -> "FakeQuery":
        return FakeQuery(self.rows[name], name)


class FakeQuery:
    def __init__(self, rows: list[dict], table_name: str) -> None:
        self.rows = rows
        self.table_name = table_name
        self.operation = "select"
        self.values: dict = {}
        self.conflict: tuple[str, ...] = ()
        self.filters: dict = {}
        self.sort_key: str | None = None
        self.desc = False

    def upsert(self, values: dict, on_conflict: str = "") -> "FakeQuery":
        self.operation = "upsert"
        self.values = values
        self.conflict = tuple(on_conflict.split(",")) if on_conflict else ()
        return self

    def update(self, values: dict) -> "FakeQuery":
        self.operation = "update"
        self.values = values
        return self

    def select(self, _: str) -> "FakeQuery":
        self.operation = "select"
        return self

    def order(self, key: str, desc: bool = False) -> "FakeQuery":
        self.sort_key = key
        self.desc = desc
        return self

    def limit(self, _: int) -> "FakeQuery":
        return self

    def eq(self, key: str, value: object) -> "FakeQuery":
        self.filters[key] = value
        return self

    def execute(self) -> SimpleNamespace:
        if self.operation == "upsert":
            row = next(
                (row for row in self.rows if self.conflict and all(row.get(k) == self.values.get(k) for k in self.conflict)),
                None,
            )
            if row is None:
                inserted = dict(self.values)
                if self.table_name == "patients":
                    inserted["id"] = f"patient-{len(self.rows) + 1}"
                self.rows.append(inserted)
                row = inserted
            else:
                row.update(self.values)
        elif self.operation == "update":
            for row in self.rows:
                if all(row.get(k) == v for k, v in self.filters.items()):
                    row.update(self.values)
        result = [row for row in self.rows if all(row.get(k) == v for k, v in self.filters.items())]
        if self.sort_key:
            result.sort(key=lambda row: row.get(self.sort_key, ""), reverse=self.desc)
        return SimpleNamespace(data=result if self.operation == "select" else [row] if self.operation == "upsert" else [])


def test_supabase_sink_writes_and_reloads_call_visit_and_events() -> None:
    async def scenario() -> None:
        client = FakeSupabase()
        sink = SupabaseEventSink("unused", "unused", client=client)
        await sink.emit(CallEvent(
            session_id="demo-1", seq=1, event_type=CallEventType.CALL_STARTED,
            payload={"caller": "+910000000000", "source": "BROWSER_SIM"},
        ))
        await sink.emit(CallEvent(
            session_id="demo-1", seq=2, event_type=CallEventType.TURN_COMPLETED,
            payload={
                "utterance_seq": 1,
                "patient": {"full_name": "Synthetic Patient"},
                "visit_values": {"bp_systolic": 150, "bp_diastolic": 95},
                "triage": {"level": "URGENT", "rules_fired": ["R3"]},
                "verification": "UNVERIFIED",
            },
        ))
        await sink.emit(CallEvent(
            session_id="demo-1", seq=3, event_type=CallEventType.CALL_ENDED,
            payload={"reason": "DROPPED", "final_verification": "INCOMPLETE"},
        ))

        assert client.rows["calls"][0]["end_reason"] == "DROPPED"
        assert client.rows["visits"][0]["verification"] == "INCOMPLETE"
        saved = await sink.load_register()
        assert len(saved["events"]) == 3
        assert saved["visits"][0]["triage"]["level"] == "URGENT"

    asyncio.run(scenario())


def test_multiple_visits_are_independent_and_only_verified_rows_get_reports() -> None:
    async def scenario() -> None:
        client = FakeSupabase()
        sink = SupabaseEventSink("unused", "unused", client=client)
        await sink.emit(CallEvent(
            session_id="multi-1", seq=1, event_type=CallEventType.CALL_STARTED,
            payload={"source": "EXOTEL", "caller": "+910000000001"}, is_simulated=False,
        ))
        for seq, visit_number, verification, name in (
            (2, 1, "VERIFIED", "Patient One"),
            (3, 2, "UNVERIFIED", "Patient Two"),
        ):
            await sink.emit(CallEvent(
                session_id="multi-1", seq=seq, event_type=CallEventType.TURN_COMPLETED,
                is_simulated=False,
                payload={
                    "visit_number": visit_number, "utterance_seq": seq - 1,
                    "patient": {"full_name": name, "village": "Rampur"},
                    "visit_values": {"gestational_weeks": 28},
                    "triage": {"level": "ROUTINE", "rules_fired": []},
                    "verification": verification,
                },
            ))
        assert len(client.rows["visits"]) == 2
        verified, pending = client.rows["visits"]
        assert verified["final_report"]["patient"]["full_name"] == "Patient One"
        assert verified["patient_id"] == "patient-1"
        assert pending["final_report"] == {}
        assert pending["verification"] == "UNVERIFIED"
        saved = await sink.load_register()
        assert len(saved["visits"]) == 2
        assert saved["calls"][0]["source"] == "EXOTEL"

    asyncio.run(scenario())


def test_history_requires_normalized_name_and_village_and_uses_guardian_clue() -> None:
    async def scenario() -> None:
        client = FakeSupabase()
        client.rows["visits"] = [
            {
                "patient_id": "patient-a", "verification": "VERIFIED", "updated_at": "2026-11-01",
                "patient": {"full_name": "Pooja Devi", "village": "Rampur", "spouse_or_guardian_name": "Ramesh Kumar"},
                "visit_values": {"gestational_weeks": 30}, "triage": {"level": "ROUTINE"},
            },
            {
                "patient_id": "patient-b", "verification": "VERIFIED", "updated_at": "2026-11-02",
                "patient": {"full_name": "Pooja Devi", "village": "Rampur", "spouse_or_guardian_name": "Mohan Lal"},
                "visit_values": {"gestational_weeks": 24}, "triage": {"level": "URGENT"},
            },
            {
                "patient_id": "patient-c", "verification": "UNVERIFIED", "updated_at": "2026-11-03",
                "patient": {"full_name": "Pooja Devi", "village": "Rampur"},
            },
        ]
        sink = SupabaseEventSink("unused", "unused", client=client)
        exact = await sink.find_patient_history({
            "full_name": " POOJA   DEVI ", "village": "rampur", "spouse_or_guardian_name": "RAMESH KUMAR",
        })
        assert [row["patient_id"] for row in exact] == ["patient-a"]
        ambiguous = await sink.find_patient_history({"full_name": "Pooja Devi", "village": "Rampur"})
        assert {row["patient_id"] for row in ambiguous} == {"patient-a", "patient-b"}
        assert await sink.find_patient_history({"full_name": "Pooja Devi", "village": "Belwa"}) == []

    asyncio.run(scenario())
