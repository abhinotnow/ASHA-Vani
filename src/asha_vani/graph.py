"""Per-call LangGraph: transcript extraction through confirmation of a visit."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from asha_vani.extraction import extract_turn
from asha_vani.schemas import (
    ConfirmationIntent,
    ExtractedVisit,
    PatientReference,
    Presence,
    SymptomCode,
    TriageLevel,
    TriageResult,
    TurnExtraction,
    ValidationResult,
)
from asha_vani.triage import evaluate_triage
from asha_vani.validation import validate_visit


class CallState(TypedDict, total=False):
    """State retained between utterances in the same call/thread."""

    utterance: str
    expected_action: str
    turn_extraction: TurnExtraction
    extracted_turn: ExtractedVisit  # compatibility for structured fixtures
    patient_turn: PatientReference
    patient: PatientReference
    visit_values: ExtractedVisit
    validation_result: ValidationResult
    triage_result: TriageResult
    next_action: str
    reply_text: str
    triage_summary: str
    verification: Literal["UNVERIFIED", "VERIFIED", "INCOMPLETE"]
    symptoms_collected: bool
    is_confirmation_turn: bool


_NUMERIC_FIELDS = ("gestational_weeks", "bp_systolic", "bp_diastolic", "hb_g_dl")
_ACTION_PROMPTS = {
    "ASK_PATIENT_NAME": "Didi, marij (patient) ka poora naam aur gaon batayein.",
    "ASK_SPOUSE_NAME": "Ji Didi, pati (husband) ka naam kya hai?",
    "ASK_WEEKS": "Theek hai. Unke pregnancy ke kitne hafte (weeks) hue hain?",
    "ASK_BP": "Ji, unka BP (blood pressure) kitna aaya hai? Systolic aur diastolic dono number batayein.",
    "ASK_HB": "Unka Hemoglobin (Hb) kitna aaya hai, aur koi takleef ya lakshan hai kya?",
    "ASK_SYMPTOMS": "Kya unhe tez sar dard, dhundhla dikhna, pet ke upar dard, bleeding, daura, paani aana, prasav dard, ya bachche ki halchal kam lagna hai?",
}


def route_after_extract(state: CallState) -> Literal["confirm", "correct", "merge"]:
    previous_action = state.get("next_action")
    extraction = TurnExtraction.model_validate(state.get("turn_extraction", {}))
    intent = extraction.confirmation_intent
    has_new_facts = bool(
        extraction.visit.model_dump(exclude_none=True, exclude_defaults=True)
        or any(extraction.patient.model_dump().values())
    )
    if previous_action == "CONFIRM_VALUES" and not has_new_facts:
        if intent is ConfirmationIntent.YES:
            return "confirm"
        if intent is ConfirmationIntent.NO:
            return "correct"
    return "merge"


def confirm_node(state: CallState) -> dict[str, Any]:
    triage = TriageResult.model_validate(state.get("triage_result", {"level": "ROUTINE"}))
    patient = PatientReference.model_validate(state.get("patient", {}))
    label = patient.full_name or "patient"
    if triage.level is TriageLevel.ROUTINE:
        reply = f"Dhanyavaad Didi. {label} ka record confirm ho gaya hai."
    else:
        reply = (
            f"Dhanyavaad Didi. {label} ka record confirm ho gaya hai. "
            f"Screening {triage.level.value} hai (rules {', '.join(triage.rules_fired)}). "
            "PHC team se turant sampark karein. Yeh screening support hai, diagnosis nahi."
        )
    return {
        "verification": "VERIFIED",
        "next_action": "COMMITTED_VERIFIED",
        "reply_text": reply,
        "is_confirmation_turn": True,
    }


def correct_node(_: CallState) -> dict[str, Any]:
    return {
        "verification": "UNVERIFIED",
        "next_action": "CORRECT_VALUES",
        "reply_text": "Theek hai Didi. Kaunsi jaankari galat hai? Sahi value dobara batayein.",
        "is_confirmation_turn": False,
    }


def merge_turn(state: CallState) -> dict[str, Any]:
    """Merge new explicit values over prior values; corrections replace old facts."""
    current = ExtractedVisit.model_validate(state.get("visit_values", {}))
    extraction = TurnExtraction.model_validate(state.get("turn_extraction", {}))
    incoming = extraction.visit
    merged_data = current.model_dump(mode="python")
    for field in _NUMERIC_FIELDS:
        value = getattr(incoming, field)
        if value is not None:
            merged_data[field] = value
    symptom_map = {report.code: report for report in current.symptoms}
    for report in incoming.symptoms:
        symptom_map[report.code] = report
    merged_data["symptoms"] = list(symptom_map.values())

    current_patient = PatientReference.model_validate(state.get("patient", {})).model_dump()
    for field, value in extraction.patient.model_dump().items():
        if value is not None:
            current_patient[field] = value

    symptoms_collected = state.get("symptoms_collected", False)
    if incoming.symptoms or state.get("next_action") == "ASK_SYMPTOMS":
        symptoms_collected = True

    return {
        "visit_values": ExtractedVisit.model_validate(merged_data).model_dump(mode="json"),
        "patient": PatientReference.model_validate(current_patient).model_dump(mode="json"),
        "symptoms_collected": symptoms_collected,
        "is_confirmation_turn": False,
        "verification": "UNVERIFIED",
    }


def validate_node(state: CallState) -> dict[str, ValidationResult]:
    visit = ExtractedVisit.model_validate(state.get("visit_values", {}))
    result = validate_visit(visit)
    return {
        "visit_values": result.values.model_dump(mode="json"),
        "validation_result": result.model_dump(mode="json"),
    }


def triage_node(state: CallState) -> dict[str, TriageResult]:
    validation = ValidationResult.model_validate(state["validation_result"])
    return {"triage_result": evaluate_triage(validation.values).model_dump(mode="json")}


def _next_missing_action(state: CallState, require_full_checklist: bool) -> str | None:
    if not require_full_checklist:
        return None
    patient = PatientReference.model_validate(state.get("patient", {}))
    visit = ExtractedVisit.model_validate(state.get("visit_values", {}))
    if not patient.full_name:
        return "ASK_PATIENT_NAME"
    if not patient.spouse_or_guardian_name:
        return "ASK_SPOUSE_NAME"
    if visit.gestational_weeks is None:
        return "ASK_WEEKS"
    if visit.bp_systolic is None or visit.bp_diastolic is None:
        return "ASK_BP"
    if visit.hb_g_dl is None:
        return "ASK_HB"
    if not state.get("symptoms_collected", False):
        return "ASK_SYMPTOMS"
    return None


def route_after_triage(state: CallState, *, require_full_checklist: bool) -> Literal["repeat", "collect", "readback"]:
    validation = ValidationResult.model_validate(state["validation_result"])
    if validation.has_errors:
        return "repeat"
    if _next_missing_action(state, require_full_checklist):
        return "collect"
    return "readback"


def ask_repeat_node(state: CallState) -> dict[str, str]:
    validation = ValidationResult.model_validate(state["validation_result"])
    first_error = next(issue for issue in validation.issues if issue.severity.value == "ERROR")
    prompt = "Kripya BP ke dono number dobara batayein." if first_error.field == "blood_pressure" else f"{first_error.message} Kripya sahi value dobara batayein."
    triage = TriageResult.model_validate(state.get("triage_result", {"level": "ROUTINE"}))
    if triage.level in (TriageLevel.URGENT, TriageLevel.CRITICAL):
        prompt = (
            f"Screening {triage.level.value} hai. Kripya PHC team se jaldi sampark karein; "
            f"yeh diagnosis nahi hai. {prompt}"
        )
    return {"next_action": "REPEAT_VALUE", "reply_text": prompt}


def collect_missing_node(state: CallState, *, require_full_checklist: bool) -> dict[str, str]:
    action = _next_missing_action(state, require_full_checklist)
    if action is None:
        return {"next_action": "CONFIRM_VALUES", "reply_text": "Kripya in jaankariyon ko confirm karein."}
    triage = TriageResult.model_validate(state.get("triage_result", {"level": "ROUTINE"}))
    prompt = _ACTION_PROMPTS[action]
    if triage.level in (TriageLevel.URGENT, TriageLevel.CRITICAL):
        prompt = (
            f"Screening {triage.level.value} hai. Kripya PHC team se jaldi sampark karein; "
            f"yeh diagnosis nahi hai. {prompt}"
        )
    return {"next_action": action, "reply_text": prompt}


def readback_node(state: CallState) -> dict[str, str]:
    visit = ExtractedVisit.model_validate(state.get("visit_values", {}))
    patient = PatientReference.model_validate(state.get("patient", {}))
    triage = TriageResult.model_validate(state["triage_result"])
    fields: list[str] = []
    if patient.full_name:
        identity = patient.full_name
        if patient.spouse_or_guardian_name:
            identity += f" W/o {patient.spouse_or_guardian_name}"
        if patient.village:
            identity += f", gaon {patient.village}"
        fields.append(identity)
    if visit.gestational_weeks is not None:
        fields.append(f"{visit.gestational_weeks} hafte")
    if visit.bp_systolic is not None and visit.bp_diastolic is not None:
        fields.append(f"BP {visit.bp_systolic} by {visit.bp_diastolic}")
    if visit.hb_g_dl is not None:
        fields.append(f"Hemoglobin {visit.hb_g_dl:g} g/dL")
    all_symptoms_absent = (
        len(visit.symptoms) == len(SymptomCode)
        and all(report.presence is Presence.ABSENT for report in visit.symptoms)
    )
    if all_symptoms_absent:
        fields.append("koi listed danger symptom nahi bataya gaya")
    else:
        for report in visit.symptoms:
            if report.presence is not Presence.UNKNOWN:
                label = report.phrase or report.code.value.replace("_", " ").lower()
                fields.append(f"{label}: {report.presence.value.lower()}")
    summary = ", ".join(fields) if fields else "abhi koi visit detail record nahi hui"
    rules = ", ".join(triage.rules_fired) or "koi danger rule trigger nahi hua"
    return {
        "next_action": "CONFIRM_VALUES",
        "reply_text": (
            f"Didi, confirm karein: {summary}. Screening {triage.level.value} hai, "
            f"{rules}. Kya yeh jaankari sahi hai? Yeh diagnosis nahi hai."
        ),
        "triage_summary": f"{triage.level.value}; rules: {', '.join(triage.rules_fired) or 'none'}; screening only.",
    }


def build_call_graph(
    checkpointer: Any | None = None,
    *,
    require_full_checklist: bool = False,
    extractor: Any = extract_turn,
) -> Any:
    """Build the turn graph. Reuse it and pass `thread_id=session_id` per call."""

    async def _extract(state: CallState) -> dict[str, Any]:
        fixture = state.get("extracted_turn")
        fixture_model = ExtractedVisit.model_validate(fixture or {})
        if any(fixture_model.model_dump(exclude_none=True, exclude_defaults=True).values()):
            result = TurnExtraction(
                patient=state.get("patient_turn", PatientReference()),
                visit=fixture_model,
            )
        else:
            result = await extractor(state.get("utterance", ""), state.get("next_action"))
        return {
            "turn_extraction": result.model_dump(mode="json"),
            "extracted_turn": ExtractedVisit().model_dump(mode="json"),
        }

    def _route_triage(state: CallState) -> Literal["repeat", "collect", "readback"]:
        return route_after_triage(state, require_full_checklist=require_full_checklist)

    def _collect(state: CallState) -> dict[str, str]:
        return collect_missing_node(state, require_full_checklist=require_full_checklist)

    builder = StateGraph(CallState)
    builder.add_node("extract", _extract)
    builder.add_node("confirm", confirm_node)
    builder.add_node("ask_correction", correct_node)
    builder.add_node("merge_turn", merge_turn)
    builder.add_node("validate", validate_node)
    builder.add_node("triage", triage_node)
    builder.add_node("ask_repeat", ask_repeat_node)
    builder.add_node("collect_missing", _collect)
    builder.add_node("readback", readback_node)
    builder.add_edge(START, "extract")
    builder.add_conditional_edges(
        "extract", route_after_extract,
        {"confirm": "confirm", "correct": "ask_correction", "merge": "merge_turn"},
    )
    builder.add_edge("confirm", END)
    builder.add_edge("ask_correction", END)
    builder.add_edge("merge_turn", "validate")
    builder.add_edge("validate", "triage")
    builder.add_conditional_edges(
        "triage", _route_triage,
        {"repeat": "ask_repeat", "collect": "collect_missing", "readback": "readback"},
    )
    builder.add_edge("ask_repeat", END)
    builder.add_edge("collect_missing", END)
    builder.add_edge("readback", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


call_graph = build_call_graph()
