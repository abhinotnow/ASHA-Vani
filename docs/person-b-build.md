# Person B: Logic, Data, and Integration Status

See the [architecture plan](ASHA-Vani-architecture-and-plan%20(1).md) for clinical rules and full product scope.

## Built

- Shared Pydantic models, hard/soft validation, and deterministic R1–R8 screening rules.
- Async transcript extraction in `src/asha_vani/extraction.py`: Groq structured output, smaller-model fallback, then conservative deterministic extraction when providers fail or are unset.
- LangGraph now owns per-call extraction state, merging/corrections, validation, triage, missing-field prompts, readback, and yes/no confirmation. `voice/turn_runner.py` is reduced to an adapter returning the existing `TurnOutput` shape.
- Minimal Supabase event sink and migration for calls, latest visit snapshot, and call events. Server credentials stay backend-only; console reload uses `/api/register`.
- Synthetic test coverage: **68 tests passed** on the current workspace.

## Still to do / coordinate

- Confirm `CallEvent` names/payload details and the adapter ownership change with Person A; current contract is a working draft.
- Add `.env` credentials and apply `supabase/migrations/202611100001_initial_demo_persistence.sql`; database-backed behavior has not been checked against a live project.
- Run the Prisma/Timbre path with real keys and tune VAD; browser speech recognition remains the fallback.
- Get the draft triage rules reviewed by a clinician and report measured latency/extraction results only.

All demo records must remain synthetic. Triage levels are screening support, not diagnoses; final referral decisions belong to PHC staff.
