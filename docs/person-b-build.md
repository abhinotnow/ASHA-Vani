# Person B: Logic, Data, and PHC Console

See the [architecture plan](ASHA-Vani-architecture-and-plan%20(1).md) for full system details. This file summarizes Person B's work and next steps.

## Built

- `src/asha_vani/schemas.py`: shared models for extracted visit values, symptoms, validation findings, and triage results.
- `src/asha_vani/validation.py`: hard/soft range checks, non-finite value rejection, and blood-pressure consistency checks. Rejected values are cleared from the returned copy.
- `src/asha_vani/triage.py`: deterministic implementation of draft rules R1–R8; returns the highest level and all matching rule IDs.
- `src/asha_vani/graph.py`: text-only turn flow; merges extracted data, validates, triages, and routes to repeat or readback using in-memory per-call state.
- `tests/test_validation.py`, `tests/test_triage.py`: 50 tests covering rule behavior and boundaries; all pass.

Clinical thresholds are drafts and need clinician review. Triage must receive the sanitized `ValidationResult.values`.

## Next

1. Add transcript extraction, missing-field collection, and yes/correction handling to the graph.
2. Agree event names, payloads, API actions, and database ownership with Person A.
3. Add persistence/outbox and build the PHC console against the agreed API.

## Team handoff

The current graph input is a structured `extracted_turn` fixture, not raw transcript text. Agree the final transcript-in/reply-out payload and event contract before voice integration.
