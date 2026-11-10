"""Per-call LangGraph: transcript extraction through confirmation of a visit."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from asha_vani.extraction import extract_turn, transliterate_to_english
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

logger = logging.getLogger(__name__)


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
    is_pregnant: bool | None
    correction_target: str | None
    symptoms_collected: bool
    is_confirmation_turn: bool
    visit_number: int
    completed_visits: list[dict[str, Any]]
    call_complete: bool
    history_checked: bool
    history_candidates: list[dict[str, Any]]
    matched_history: dict[str, Any]
    history_candidate_index: int
    patient_match_status: Literal["EXISTING", "NEW", "UNAVAILABLE"]
    matched_patient_id: str | None
    changed_readback: list[str]


_NUMERIC_FIELDS = ("gestational_weeks", "bp_systolic", "bp_diastolic", "hb_g_dl")
_NON_PREGNANT_RE = re.compile(
    r"(?:\b(?:pregnant\s+nahi|garbhvati\s+nahi|garbhini\s+nahi|not\s+pregnant|no\s+pregnancy|"
    r"postnatal|post\s+natal|pnc|delivery\s+ho\s+gayi|general\s+checkup|normal\s+checkup|"
    r"0\s*(?:weeks?|hafte|hafta|saptah)|zero\s*(?:weeks?|hafte|hafta|saptah)|shunya\s*(?:weeks?|hafte|hafta|saptah))\b|"
    r"गर्भवती\s+नहीं|0\s*(?:हफ्ते|हफ़्ते|सप्ताह)|शून्य\s*(?:हफ्ते|हफ़्ते|सप्ताह))",
    re.IGNORECASE,
)
_SYMPTOM_HINDI_LABELS: dict[SymptomCode, str] = {
    SymptomCode.SEVERE_HEADACHE: "tez sar dard",
    SymptomCode.BLURRED_VISION: "dhundhla dikhna",
    SymptomCode.EPIGASTRIC_PAIN: "pet mein dard",
    SymptomCode.CONVULSION: "daura",
    SymptomCode.UNCONSCIOUS: "behoshi",
    SymptomCode.VAGINAL_BLEEDING: "bleeding",
    SymptomCode.LABOUR_PAINS: "prasav dard",
    SymptomCode.LEAKING_FLUID: "paani aana",
    SymptomCode.REDUCED_FETAL_MOVEMENT: "bachche ki halchal kam",
    SymptomCode.SWELLING: "soojan",
}
_ACTION_PROMPTS = {
    "ASK_PATIENT_NAME": "Didi, marij (patient) ka poora naam aur gaon batayein.",
    "ASK_SPOUSE_NAME": "Ji Didi, pati (husband) ka naam kya hai?",
    "ASK_WEEKS": "Theek hai. Unke pregnancy ke kitne hafte (weeks) hue hain? Agar garbhvati nahi hain toh zero batayein.",
    "ASK_BP": "Ji, unka BP (blood pressure) kitna aaya hai? Systolic aur diastolic dono number batayein.",
    "ASK_HB": "Unka Hemoglobin (Hb) kitna aaya hai, aur koi takleef ya lakshan hai kya?",
    "ASK_SYMPTOMS": "Kya unhe pet mein dard, tez sar dard, dhundhla dikhna, bleeding, daura, soojan, paani aana, ya bachche ki halchal kam lagna hai?",
}


def route_after_extract(state: CallState) -> Literal[
    "confirm", "correct", "merge", "new_visit", "finish_call", "continue_prompt",
    "history_yes", "history_no", "history_prompt", "read_history", "skip_history", "history_repeat"
]:
    previous_action = state.get("next_action")
    extraction = TurnExtraction.model_validate(state.get("turn_extraction", {}))
    intent = extraction.confirmation_intent
    has_new_facts = bool(
        extraction.visit.model_dump(exclude_none=True, exclude_defaults=True)
        or any(extraction.patient.model_dump().values())
    )
    if previous_action == "CONFIRM_PATIENT_MATCH":
        if intent is ConfirmationIntent.YES:
            return "history_yes"
        if intent is ConfirmationIntent.NO:
            return "history_no"
        return "history_prompt"
    if previous_action == "ASK_READ_PRIOR_RECORD":
        if intent is ConfirmationIntent.YES:
            return "read_history"
        if intent is ConfirmationIntent.NO:
            return "skip_history"
        return "history_repeat"
    if previous_action == "ASK_ANOTHER_PATIENT":
        if intent is ConfirmationIntent.NO:
            return "finish_call"
        if intent is ConfirmationIntent.YES or has_new_facts:
            return "new_visit"
        return "continue_prompt"
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
        reply = f"Dhanyavaad Didi. {label} ka record confirm ho gaya hai. Kya aap kisi aur mahila ka record dena chahti hain?"
    else:
        reply = (
            f"Dhanyavaad Didi. {label} ka record confirm ho gaya hai. "
            f"Screening {triage.level.value} hai, kripya PHC team se turant sampark karein. "
            "Kya aap kisi aur mahila ka record dena chahti hain?"
        )
    return {
        "verification": "VERIFIED",
        "next_action": "ASK_ANOTHER_PATIENT",
        "reply_text": reply,
        "is_confirmation_turn": True,
    }


def start_next_visit_node(state: CallState) -> dict[str, Any]:
    """Start a new visit within the current phone call after explicit yes."""
    completed = list(state.get("completed_visits", []))
    completed.append({
        "visit_number": state.get("visit_number", 1),
        "patient": PatientReference.model_validate(state.get("patient", {})).model_dump(mode="json"),
        "visit_values": ExtractedVisit.model_validate(state.get("visit_values", {})).model_dump(mode="json"),
        "triage": TriageResult.model_validate(state.get("triage_result", {"level": "ROUTINE"})).model_dump(mode="json"),
        "verification": state.get("verification", "UNVERIFIED"),
    })
    next_number = int(state.get("visit_number", 1)) + 1
    extraction = TurnExtraction.model_validate(state.get("turn_extraction", {}))
    return {
        "completed_visits": completed,
        "visit_number": next_number,
        "patient": extraction.patient.model_dump(mode="json"),
        "visit_values": extraction.visit.model_dump(mode="json"),
        "validation_result": validate_visit(ExtractedVisit()).model_dump(mode="json"),
        "triage_result": TriageResult(level=TriageLevel.ROUTINE).model_dump(mode="json"),
        "is_pregnant": None,
        "symptoms_collected": False,
        "verification": "UNVERIFIED",
        "history_checked": False,
        "history_candidates": [],
        "history_candidate_index": 0,
        "matched_history": {},
        "matched_patient_id": None,
        "patient_match_status": "UNAVAILABLE",
        "next_action": "ASK_PATIENT_NAME",
        "reply_text": "Theek hai Didi. Ab agle marij ka poora naam aur gaon batayein.",
        "is_confirmation_turn": False,
    }


def finish_call_node(state: CallState) -> dict[str, Any]:
    """Close politely after the ASHA says there is no additional patient."""
    return {
        "next_action": "END_CALL",
        "reply_text": "Dhanyavaad Didi. ASHA-Vani call ab band ho rahi hai. Namaste.",
        "call_complete": True,
        "is_confirmation_turn": False,
    }


def continue_prompt_node(_: CallState) -> dict[str, str]:
    return {
        "next_action": "ASK_ANOTHER_PATIENT",
        "reply_text": "Didi, kya aap kisi aur garbhavati mahila ka record dena chahti hain? Haan ya nahi batayein.",
    }


def history_prompt_node(state: CallState) -> dict[str, str]:
    patient = PatientReference.model_validate(state.get("patient", {}))
    candidates = state.get("history_candidates") or [{}]
    index = min(state.get("history_candidate_index", 0), len(candidates) - 1)
    candidate = candidates[index]
    prior = candidate.get("patient") or {}
    name = prior.get("full_name") or patient.full_name or "is marij"
    village = prior.get("village") or patient.village or "aapke gaon"
    return {
        "next_action": "CONFIRM_PATIENT_MATCH",
        "reply_text": (
            f"Mujhe {village} ki {name} ka purana record mila hai. "
            "Kya yeh wahi patient hain? Haan ya nahi batayein."
        ),
    }


def history_yes_node(state: CallState) -> dict[str, Any]:
    candidates = state.get("history_candidates") or [{}]
    index = min(state.get("history_candidate_index", 0), len(candidates) - 1)
    candidate = candidates[index]
    return {
        "matched_history": candidate,
        "matched_patient_id": candidate.get("patient_id"),
        "patient_match_status": "EXISTING",
        "history_checked": True,
        "next_action": "ASK_READ_PRIOR_RECORD",
        "reply_text": "Theek hai. Kya aap unka pichhla visit summary sunna chahti hain? Haan ya nahi batayein.",
    }


def history_no_node(state: CallState) -> dict[str, Any]:
    next_index = state.get("history_candidate_index", 0) + 1
    candidates = state.get("history_candidates") or []
    if next_index < len(candidates):
        candidate = candidates[next_index]
        prior = candidate.get("patient") or {}
        return {
            "history_candidate_index": next_index,
            "next_action": "CONFIRM_PATIENT_MATCH",
            "reply_text": (
                f"Theek hai. Ek aur purana record mila: {prior.get('full_name', 'patient')}, "
                f"gaon {prior.get('village', 'unknown')}, "
                f"pati/guardian {prior.get('spouse_or_guardian_name', 'unknown')}. Kya yeh wahi patient hain?"
            ),
        }
    return {
        "history_checked": True,
        "history_candidates": [],
        "matched_history": {},
        "matched_patient_id": None,
        "patient_match_status": "NEW",
        "next_action": "CONTINUE_INTAKE",
        "reply_text": "Theek hai, main ise alag patient ke roop mein darj karungi.",
    }


def read_history_node(state: CallState, *, require_full_checklist: bool) -> dict[str, str]:
    record = state.get("matched_history", {})
    report = record.get("final_report") or {}
    patient = report.get("patient") or record.get("patient") or {}
    values = report.get("visit_values") or record.get("visit_values") or {}
    triage = report.get("triage") or record.get("triage") or {}
    details = []
    if values.get("gestational_weeks") is not None:
        details.append(f"{values['gestational_weeks']} hafte")
    if values.get("bp_systolic") is not None and values.get("bp_diastolic") is not None:
        details.append(f"BP {values['bp_systolic']} by {values['bp_diastolic']}")
    if values.get("hb_g_dl") is not None:
        details.append(f"Hemoglobin {values['hb_g_dl']} g/dL")
    details.append(f"screening {triage.get('level', 'unknown')}")
    action = _next_missing_action(state, require_full_checklist) or "CONFIRM_VALUES"
    prompt = _ACTION_PROMPTS.get(action, "Ab naye visit ki jaankari confirm karein.")
    name = patient.get("full_name") or "patient"
    return {
        "next_action": action,
        "reply_text": f"Pichhle visit mein {name} ke liye {', '.join(details)} darj tha. Yeh sirf purana record hai. {prompt}",
    }


def history_repeat_node(_: CallState) -> dict[str, str]:
    return {
        "next_action": "ASK_READ_PRIOR_RECORD",
        "reply_text": "Didi, kya aap pichhla visit summary sunna chahti hain? Haan ya nahi batayein.",
    }


def correct_node(state: CallState) -> dict[str, Any]:
    roman_utt = transliterate_to_english(state.get("utterance", "")).lower()
    target: str | None = None
    reply = "Theek hai Didi. Kaunsi jaankari galat hai? Sahi value dobara batayein."
    if re.search(r"\b(pati|husband|spouse)\b", roman_utt):
        target = "ASK_SPOUSE_NAME"
        reply = "Theek hai Didi. Pati ka sahi naam batayein."
    elif re.search(r"\b(marij|patient|patni|mahila|aurat|naam)\b", roman_utt):
        target = "ASK_PATIENT_NAME"
        reply = "Theek hai Didi. Marij ka sahi naam batayein."
    elif re.search(r"\b(bp|blood pressure)\b", roman_utt):
        target = "ASK_BP"
        reply = "Theek hai Didi. Sahi BP ke dono number batayein."
    elif re.search(r"\b(hafte|hafta|weeks?)\b", roman_utt):
        target = "ASK_WEEKS"
        reply = "Theek hai Didi. Pregnancy ke sahi hafte batayein."
    elif re.search(r"\b(hb|hemoglobin|khoon)\b", roman_utt):
        target = "ASK_HB"
        reply = "Theek hai Didi. Sahi Hemoglobin (Hb) batayein."
    return {
        "verification": "UNVERIFIED",
        "next_action": "CORRECT_VALUES",
        "correction_target": target,
        "reply_text": reply,
        "is_confirmation_turn": False,
    }


def merge_turn(state: CallState) -> dict[str, Any]:
    """Merge new explicit values over prior values; corrections replace old facts."""
    current = ExtractedVisit.model_validate(state.get("visit_values", {}))
    extraction = TurnExtraction.model_validate(state.get("turn_extraction", {}))
    incoming = extraction.visit
    utterance = state.get("utterance", "")
    roman_utt = transliterate_to_english(utterance)
    prev_action = state.get("correction_target") or state.get("next_action")
    is_pregnant = state.get("is_pregnant")

    non_pregnant_stated = bool(
        incoming.gestational_weeks == 0
        or _NON_PREGNANT_RE.search(utterance)
        or (
            prev_action == "ASK_WEEKS"
            and re.search(r"^\s*(?:0|zero|shunya|nahi|nahin|no)\b|^\s*(?:नहीं|शून्य)\b", utterance.strip(), re.I)
        )
    )
    if non_pregnant_stated:
        is_pregnant = False

    merged_data = current.model_dump(mode="python")
    for field in _NUMERIC_FIELDS:
        value = getattr(incoming, field)
        if field == "gestational_weeks" and (value == 0 or non_pregnant_stated):
            continue
        if value is not None:
            merged_data[field] = value
            if field == "gestational_weeks" and value > 0:
                is_pregnant = True
    if is_pregnant is False:
        merged_data["gestational_weeks"] = None

    symptom_map = {report.code: report for report in current.symptoms}
    for report in incoming.symptoms:
        symptom_map[report.code] = report
    merged_data["symptoms"] = list(symptom_map.values())

    current_patient = PatientReference.model_validate(state.get("patient", {})).model_dump()
    incoming_patient = extraction.patient.model_dump()
    mentions_spouse = bool(re.search(r"\b(pati|husband|spouse)\b", roman_utt, re.I))
    mentions_patient = bool(re.search(r"\b(marij|patient|mahila|aurat)\b", roman_utt, re.I))
    if (
        (prev_action == "ASK_SPOUSE_NAME" or (mentions_spouse and not mentions_patient and "patni" not in roman_utt.lower()))
        and current_patient.get("full_name")
    ):
        if incoming_patient.get("full_name") and not incoming_patient.get("spouse_or_guardian_name"):
            incoming_patient["spouse_or_guardian_name"] = incoming_patient["full_name"]
        if not mentions_patient:
            incoming_patient["full_name"] = None

    for field, value in incoming_patient.items():
        if value is not None:
            current_patient[field] = value

    is_correction = state.get("next_action") in {"CONFIRM_VALUES", "CORRECT_VALUES"}
    changed_readback: list[str] = []
    if is_correction:
        if incoming_patient.get("full_name"):
            changed_readback.append(f"marij ka naam {current_patient['full_name']}")
        if incoming_patient.get("spouse_or_guardian_name"):
            changed_readback.append(f"pati ka naam {current_patient['spouse_or_guardian_name']}")
        if incoming_patient.get("village"):
            changed_readback.append(f"gaon {current_patient['village']}")
        if non_pregnant_stated:
            changed_readback.append("Non-Pregnant Checkup")
        elif incoming.gestational_weeks is not None and merged_data.get("gestational_weeks") is not None:
            changed_readback.append(f"{merged_data['gestational_weeks']} hafte")
        if (incoming.bp_systolic is not None or incoming.bp_diastolic is not None) and (
            merged_data.get("bp_systolic") is not None and merged_data.get("bp_diastolic") is not None
        ):
            changed_readback.append(f"BP {merged_data['bp_systolic']} by {merged_data['bp_diastolic']}")
        if incoming.hb_g_dl is not None and merged_data.get("hb_g_dl") is not None:
            changed_readback.append(f"Hemoglobin {merged_data['hb_g_dl']:g} g/dL")
        for report in incoming.symptoms:
            if report.presence is Presence.PRESENT:
                label = _SYMPTOM_HINDI_LABELS.get(report.code) or report.phrase or report.code.value.replace("_", " ").lower()
                changed_readback.append(label)

    symptoms_collected = state.get("symptoms_collected", False)
    if incoming.symptoms or prev_action == "ASK_SYMPTOMS":
        symptoms_collected = True

    return {
        "visit_values": ExtractedVisit.model_validate(merged_data).model_dump(mode="json"),
        "patient": PatientReference.model_validate(current_patient).model_dump(mode="json"),
        "is_pregnant": is_pregnant,
        "correction_target": None,
        "changed_readback": changed_readback,
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
    if visit.gestational_weeks is None and state.get("is_pregnant") is not False:
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
        prompt = f"Screening {triage.level.value} hai, kripya PHC team se jaldi sampark karein. {prompt}"
    return {"next_action": "REPEAT_VALUE", "reply_text": prompt}


def collect_missing_node(state: CallState, *, require_full_checklist: bool) -> dict[str, str]:
    action = _next_missing_action(state, require_full_checklist)
    if action is None:
        return {"next_action": "CONFIRM_VALUES", "reply_text": "Kripya in jaankariyon ko confirm karein."}
    prompt = _ACTION_PROMPTS[action]
    if action == "ASK_BP" and state.get("is_pregnant") is False and state.get("next_action") == "ASK_WEEKS":
        prompt = (
            "Ji Didi, Non-Pregnant checkup note ho gaya. "
            "Unka BP (blood pressure) kitna aaya hai? Systolic aur diastolic dono number batayein."
        )
    return {"next_action": action, "reply_text": prompt}


def readback_node(state: CallState) -> dict[str, str]:
    visit = ExtractedVisit.model_validate(state.get("visit_values", {}))
    patient = PatientReference.model_validate(state.get("patient", {}))
    triage = TriageResult.model_validate(state["triage_result"])
    fields: list[str] = []
    if patient.full_name:
        identity = f"marij ka naam {patient.full_name}"
        if patient.spouse_or_guardian_name:
            identity += f", pati ka naam {patient.spouse_or_guardian_name}"
        if patient.village:
            identity += f", gaon {patient.village}"
        fields.append(identity)
    if visit.gestational_weeks is not None:
        fields.append(f"{visit.gestational_weeks} hafte")
    elif state.get("is_pregnant") is False:
        fields.append("Non-Pregnant Checkup")
    if visit.bp_systolic is not None and visit.bp_diastolic is not None:
        fields.append(f"BP {visit.bp_systolic} by {visit.bp_diastolic}")
    if visit.hb_g_dl is not None:
        fields.append(f"Hemoglobin {visit.hb_g_dl:g} g/dL")
    all_symptoms_absent = (
        len(visit.symptoms) == len(SymptomCode)
        and all(report.presence is Presence.ABSENT for report in visit.symptoms)
    )
    if all_symptoms_absent:
        fields.append("koi danger lakshan nahi")
    else:
        for report in visit.symptoms:
            if report.presence is Presence.PRESENT:
                label = _SYMPTOM_HINDI_LABELS.get(report.code) or report.phrase or report.code.value.replace("_", " ").lower()
                fields.append(label)
            elif report.presence is Presence.ABSENT:
                label = _SYMPTOM_HINDI_LABELS.get(report.code) or report.phrase or report.code.value.replace("_", " ").lower()
                fields.append(f"{label} nahi hai")
    summary = ", ".join(fields) if fields else "abhi koi visit detail record nahi hui"
    changed_readback = state.get("changed_readback") or []
    if changed_readback:
        changed_summary = ", ".join(changed_readback)
        reply_text = (
            f"Theek hai, badal diya: {changed_summary}. Screening {triage.level.value} hai. "
            "Kya yeh jaankari sahi hai?"
        )
    else:
        reply_text = (
            f"Didi, confirm karein: {summary}. Screening {triage.level.value} hai. "
            "Kya yeh jaankari sahi hai?"
        )
    return {
        "next_action": "CONFIRM_VALUES",
        "changed_readback": [],
        "reply_text": reply_text,
        "triage_summary": f"{triage.level.value}; rules: {', '.join(triage.rules_fired) or 'none'}; screening only.",
    }


def build_call_graph(
    checkpointer: Any | None = None,
    *,
    require_full_checklist: bool = False,
    extractor: Any = extract_turn,
    history_lookup: Any | None = None,
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
            effective_action = state.get("correction_target") or state.get("next_action")
            result = await extractor(state.get("utterance", ""), effective_action)
        return {
            "turn_extraction": result.model_dump(mode="json"),
            "extracted_turn": ExtractedVisit().model_dump(mode="json"),
        }

    def _route_triage(state: CallState) -> Literal["repeat", "collect", "readback"]:
        return route_after_triage(state, require_full_checklist=require_full_checklist)

    def _collect(state: CallState) -> dict[str, str]:
        return collect_missing_node(state, require_full_checklist=require_full_checklist)

    async def _lookup_history(state: CallState) -> dict[str, Any]:
        patient = PatientReference.model_validate(state.get("patient", {}))
        if state.get("history_checked") or not patient.full_name or not patient.village:
            return {}
        if history_lookup is None:
            return {"history_checked": True, "patient_match_status": "UNAVAILABLE"}
        try:
            candidates = await history_lookup(patient.model_dump(mode="json"))
        except Exception:
            logger.exception("Patient history lookup failed; continuing intake without a match")
            return {"history_checked": True, "patient_match_status": "UNAVAILABLE"}
        if not candidates:
            return {
                "history_candidates": [],
                "history_candidate_index": 0,
                "history_checked": True,
                "patient_match_status": "NEW",
            }
        top_candidate = candidates[0]
        return {
            "history_candidates": candidates,
            "history_candidate_index": 0,
            "matched_history": top_candidate,
            "matched_patient_id": top_candidate.get("patient_id"),
            "history_checked": True,
            "patient_match_status": "EXISTING",
        }

    def _route_history(_: CallState) -> Literal["prompt", "validate"]:
        return "validate"

    def _route_history_no(state: CallState) -> Literal["another", "resume"]:
        return "another" if state.get("history_candidate_index", 0) < len(state.get("history_candidates", [])) else "resume"

    builder = StateGraph(CallState)
    builder.add_node("extract", _extract)
    builder.add_node("confirm", confirm_node)
    builder.add_node("history_lookup", _lookup_history)
    builder.add_node("history_prompt", history_prompt_node)
    builder.add_node("history_yes", history_yes_node)
    builder.add_node("history_no", history_no_node)
    builder.add_node("skip_history", lambda _: {"history_checked": True, "history_candidates": [], "matched_history": {}, "matched_patient_id": None, "next_action": "CONTINUE_INTAKE"})
    builder.add_node("read_history", lambda state: read_history_node(state, require_full_checklist=require_full_checklist))
    builder.add_node("history_repeat", history_repeat_node)
    builder.add_node("start_next_visit", start_next_visit_node)
    builder.add_node("finish_call", finish_call_node)
    builder.add_node("continue_prompt", continue_prompt_node)
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
        {
            "confirm": "confirm", "correct": "ask_correction", "merge": "merge_turn",
            "new_visit": "start_next_visit", "finish_call": "finish_call",
            "continue_prompt": "continue_prompt",
            "history_yes": "history_yes", "history_no": "history_no",
            "history_prompt": "history_prompt", "read_history": "read_history",
            "skip_history": "skip_history", "history_repeat": "history_repeat",
        },
    )
    builder.add_edge("confirm", END)
    builder.add_edge("start_next_visit", END)
    builder.add_edge("finish_call", END)
    builder.add_edge("continue_prompt", END)
    builder.add_edge("ask_correction", END)
    builder.add_edge("merge_turn", "history_lookup")
    builder.add_conditional_edges("history_lookup", _route_history, {"prompt": "history_prompt", "validate": "validate"})
    builder.add_edge("history_prompt", END)
    builder.add_edge("history_yes", END)
    builder.add_conditional_edges("history_no", _route_history_no, {"another": END, "resume": "merge_turn"})
    builder.add_edge("skip_history", "merge_turn")
    builder.add_edge("read_history", END)
    builder.add_edge("history_repeat", END)
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
