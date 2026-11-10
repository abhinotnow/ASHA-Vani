"""Thin adapter from a completed voice transcript to the LangGraph call flow."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from pydantic import Field

from asha_vani.graph import build_call_graph
from asha_vani.schemas import (
    ExtractedVisit,
    PatientReference,
    ProjectModel,
    TriageResult,
    TurnExtraction,
    ValidationResult,
)


class VerificationStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    INCOMPLETE = "INCOMPLETE"


class TurnOutput(ProjectModel):
    """Stable result shape consumed by CallSession and turn-level events."""

    session_id: str
    utterance_seq: int
    visit_number: int = 1
    matched_patient_id: str | None = None
    completed_visits: list[dict[str, Any]] = Field(default_factory=list)
    call_complete: bool = False
    transcript: str
    patient: PatientReference = Field(default_factory=PatientReference)
    extracted_turn: ExtractedVisit = Field(default_factory=ExtractedVisit)
    visit_values: ExtractedVisit = Field(default_factory=ExtractedVisit)
    validation_result: ValidationResult
    triage_result: TriageResult
    verification: VerificationStatus
    next_action: str
    reply_text: str
    triage_summary: str = ""
    is_confirmation_turn: bool = False


class TurnRunner(Protocol):
    async def run_turn(self, session_id: str, utterance_seq: int, transcript: str) -> TurnOutput: ...


class LangGraphTurnRunner:
    """Invoke one compiled per-call graph; all conversation state lives in its checkpoint."""

    def __init__(
        self,
        graph: Any | None = None,
        *,
        require_full_checklist: bool = False,
        history_lookup: Any | None = None,
    ) -> None:
        self._graph = graph or build_call_graph(
            require_full_checklist=require_full_checklist,
            history_lookup=history_lookup,
        )

    async def run_turn(self, session_id: str, utterance_seq: int, transcript: str) -> TurnOutput:
        state = await self._graph.ainvoke(
            {"utterance": transcript},
            config={"configurable": {"thread_id": session_id}},
        )
        extraction = TurnExtraction.model_validate(state.get("turn_extraction", {}))
        return TurnOutput(
            session_id=session_id,
            utterance_seq=utterance_seq,
            visit_number=int(state.get("visit_number", 1)),
            matched_patient_id=state.get("matched_patient_id"),
            completed_visits=list(state.get("completed_visits", [])),
            call_complete=bool(state.get("call_complete", False)),
            transcript=transcript,
            patient=PatientReference.model_validate(state.get("patient", {})),
            extracted_turn=extraction.visit,
            visit_values=ExtractedVisit.model_validate(state.get("visit_values", {})),
            validation_result=ValidationResult.model_validate(state["validation_result"]),
            triage_result=TriageResult.model_validate(state["triage_result"]),
            verification=VerificationStatus(state.get("verification", "UNVERIFIED")),
            next_action=str(state.get("next_action", "")),
            reply_text=str(state.get("reply_text", "")),
            triage_summary=str(state.get("triage_summary", "")),
            is_confirmation_turn=bool(state.get("is_confirmation_turn", False)),
        )
