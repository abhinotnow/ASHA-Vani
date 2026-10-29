"""First text-only conversation graph for one ASHA-Vani call.

Speech recognition and LLM extraction are not connected yet. For now, each
invocation receives a structured ``extracted_turn`` fixture.
"""

from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from asha_vani.schemas import (
    ExtractedVisit,
    Presence,
    TriageResult,
    ValidationResult,
)
from asha_vani.triage import evaluate_triage
from asha_vani.validation import validate_visit


class CallState(TypedDict, total=False):
    """Shared state for a call; nodes update only the keys they produce."""

    utterance: str
    extracted_turn: ExtractedVisit
    visit_values: ExtractedVisit
    validation_result: ValidationResult
    triage_result: TriageResult
    next_action: Literal["REPEAT_VALUE", "CONFIRM_VALUES"]
    reply_text: str
    triage_summary: str


_NUMERIC_FIELDS = (
    "gestational_weeks",
    "bp_systolic",
    "bp_diastolic",
    "hb_g_dl",
)


def merge_turn(state: CallState) -> dict[str, Any]:
    """Add this turn's extracted details to the accumulated visit.

    A missing number means "not mentioned in this turn". A symptom report
    replaces the earlier report with the same code, so a correction wins.
    """

    current = ExtractedVisit.model_validate(state.get("visit_values", {}))
    incoming = ExtractedVisit.model_validate(state.get("extracted_turn", {}))
    merged_data = current.model_dump(mode="python")

    for field in _NUMERIC_FIELDS:
        new_value = getattr(incoming, field)
        if new_value is not None:
            merged_data[field] = new_value

    symptoms_by_code = {report.code: report for report in current.symptoms}
    for report in incoming.symptoms:
        symptoms_by_code[report.code] = report
    merged_data["symptoms"] = list(symptoms_by_code.values())

    merged_visit = ExtractedVisit.model_validate(merged_data)
    return {"visit_values": merged_visit, "extracted_turn": ExtractedVisit()}


def validate_node(state: CallState) -> dict[str, ValidationResult | ExtractedVisit]:
    """Validate the accumulated visit and retain only usable values."""

    visit = ExtractedVisit.model_validate(state.get("visit_values", {}))
    result = validate_visit(visit)
    return {"visit_values": result.values, "validation_result": result}


def triage_node(state: CallState) -> dict[str, TriageResult]:
    """Screen sanitized values, even when another value needs repeating."""

    validation = ValidationResult.model_validate(state["validation_result"])
    return {"triage_result": evaluate_triage(validation.values)}


def route_after_triage(state: CallState) -> Literal["repeat", "readback"]:
    """Route hard errors to a repeat; otherwise read back captured details."""

    validation = ValidationResult.model_validate(state["validation_result"])
    return "repeat" if validation.has_errors else "readback"


def ask_repeat_node(state: CallState) -> dict[str, str]:
    """Ask about the first hard error; resolve further errors on later turns."""

    validation = ValidationResult.model_validate(state["validation_result"])
    first_error = next(
        issue for issue in validation.issues if issue.severity.value == "ERROR"
    )
    return {
        "next_action": "REPEAT_VALUE",
        "reply_text": (
            "I may have heard that incorrectly. Please repeat it. "
            f"{first_error.message}"
        ),
    }


def readback_node(state: CallState) -> dict[str, str]:
    """Create a simple English readback for this development-stage graph."""

    visit = ExtractedVisit.model_validate(state.get("visit_values", {}))
    details: list[str] = []

    if visit.gestational_weeks is not None:
        details.append(f"pregnancy duration: {visit.gestational_weeks} weeks")

    if visit.bp_systolic is not None:
        details.append(f"systolic blood pressure: {visit.bp_systolic}")
    if visit.bp_diastolic is not None:
        details.append(f"diastolic blood pressure: {visit.bp_diastolic}")

    if visit.hb_g_dl is not None:
        details.append(f"hemoglobin: {visit.hb_g_dl:g} g/dL")

    for report in visit.symptoms:
        if report.presence is Presence.UNKNOWN:
            continue
        name = report.phrase or report.code.value.replace("_", " ").lower()
        details.append(f"{name}: {report.presence.value.lower()}")

    triage = TriageResult.model_validate(state["triage_result"])
    fired_rules = ", ".join(triage.rules_fired) or "none"
    if details:
        reply = "Please confirm these details: " + "; ".join(details) + "."
    else:
        reply = (
            "I have not captured visit details yet. "
            "Please share the available measurements or symptoms."
        )

    return {
        "next_action": "CONFIRM_VALUES",
        "reply_text": reply,
        "triage_summary": (
            f"{triage.level.value} (rules: {fired_rules}); "
            "screening result, not a diagnosis."
        ),
    }


def build_call_graph(checkpointer: Any | None = None) -> Any:
    """Build and compile the graph; reuse a thread ID to resume a call."""

    builder = StateGraph(CallState)
    builder.add_node("merge_turn", merge_turn)
    builder.add_node("validate", validate_node)
    builder.add_node("triage", triage_node)
    builder.add_node("ask_repeat", ask_repeat_node)
    builder.add_node("readback", readback_node)

    builder.add_edge(START, "merge_turn")
    builder.add_edge("merge_turn", "validate")
    builder.add_edge("validate", "triage")
    builder.add_conditional_edges(
        "triage",
        route_after_triage,
        {"repeat": "ask_repeat", "readback": "readback"},
    )
    builder.add_edge("ask_repeat", END)
    builder.add_edge("readback", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


call_graph = build_call_graph()
