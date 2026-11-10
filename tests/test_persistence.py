from __future__ import annotations

import asyncio
from types import SimpleNamespace

from asha_vani.persistence import SupabaseEventSink
from asha_vani.voice.events import CallEvent, CallEventType


class FakeSupabase:
    def __init__(self) -> None:
        self.rows: dict[str, list[dict]] = {"calls": [], "visits": [], "call_events": []}

    def table(self, name: str) -> "FakeQuery":
        return FakeQuery(self.rows[name])


class FakeQuery:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
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
                self.rows.append(dict(self.values))
            else:
                row.update(self.values)
        elif self.operation == "update":
            for row in self.rows:
                if all(row.get(k) == v for k, v in self.filters.items()):
                    row.update(self.values)
        result = [row for row in self.rows if all(row.get(k) == v for k, v in self.filters.items())]
        if self.sort_key:
            result.sort(key=lambda row: row.get(self.sort_key, ""), reverse=self.desc)
        return SimpleNamespace(data=result)


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
