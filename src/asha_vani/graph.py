"""Per-call LangGraph: transcript extraction through confirmation of a visit."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from asha_vani.extraction import extract_turn, transliterate_to_english
from asha_vani.language.locale import DEFAULT_LANGUAGE, lexicon, readback_field, symptom_label, t
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
    language: str
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
def _keyword_regex(lang: str, category: str) -> re.Pattern[str]:
    terms = list(lexicon(lang).get("keywords", {}).get(category, []))
    if lang != DEFAULT_LANGUAGE:
        terms.extend(lexicon(DEFAULT_LANGUAGE).get("keywords", {}).get(category, []))
    escaped = "|".join(re.escape(term) for term in dict.fromkeys(terms) if term)
    return re.compile(rf"(?:{escaped})", re.I) if escaped else re.compile(r"(?!)")


def _state_lang(state: CallState) -> str:
    return str(state.get("language") or DEFAULT_LANGUAGE)


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
    lang = _state_lang(state)
    triage = TriageResult.model_validate(state.get("triage_result", {"level": "ROUTINE"}))
    patient = PatientReference.model_validate(state.get("patient", {}))
    label = patient.full_name or "patient"
    if triage.level is TriageLevel.ROUTINE:
        reply = t(lang, "confirm_routine", label=label)
    else:
        reply = t(lang, "confirm_escalate", label=label, level=triage.level.value)
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
        "reply_text": t(_state_lang(state), "next_patient"),
        "is_confirmation_turn": False,
    }


def finish_call_node(state: CallState) -> dict[str, Any]:
    """Close politely after the ASHA says there is no additional patient."""
    return {
        "next_action": "END_CALL",
        "reply_text": t(_state_lang(state), "finish_call"),
        "call_complete": True,
        "is_confirmation_turn": False,
    }


def continue_prompt_node(state: CallState) -> dict[str, str]:
    return {
        "next_action": "ASK_ANOTHER_PATIENT",
        "reply_text": t(_state_lang(state), "continue_prompt"),
    }


def history_prompt_node(state: CallState) -> dict[str, str]:
    lang = _state_lang(state)
    patient = PatientReference.model_validate(state.get("patient", {}))
    candidates = state.get("history_candidates") or [{}]
    index = min(state.get("history_candidate_index", 0), len(candidates) - 1)
    candidate = candidates[index]
    prior = candidate.get("patient") or {}
    name = prior.get("full_name") or patient.full_name or "patient"
    village = prior.get("village") or patient.village or "village"
    return {
        "next_action": "CONFIRM_PATIENT_MATCH",
        "reply_text": t(lang, "history_match", village=village, name=name),
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
        "reply_text": t(_state_lang(state), "history_yes"),
    }


def history_no_node(state: CallState) -> dict[str, Any]:
    lang = _state_lang(state)
    next_index = state.get("history_candidate_index", 0) + 1
    candidates = state.get("history_candidates") or []
    if next_index < len(candidates):
        candidate = candidates[next_index]
        prior = candidate.get("patient") or {}
        return {
            "history_candidate_index": next_index,
            "next_action": "CONFIRM_PATIENT_MATCH",
            "reply_text": t(
                lang,
                "history_another",
                name=prior.get("full_name", "patient"),
                village=prior.get("village", "unknown"),
                spouse=prior.get("spouse_or_guardian_name", "unknown"),
            ),
        }
    return {
        "history_checked": True,
        "history_candidates": [],
        "matched_history": {},
        "matched_patient_id": None,
        "patient_match_status": "NEW",
        "next_action": "CONTINUE_INTAKE",
        "reply_text": t(lang, "history_no_more"),
    }


def read_history_node(state: CallState, *, require_full_checklist: bool) -> dict[str, str]:
    lang = _state_lang(state)
    record = state.get("matched_history", {})
    report = record.get("final_report") or {}
    patient = report.get("patient") or record.get("patient") or {}
    values = report.get("visit_values") or record.get("visit_values") or {}
    triage = report.get("triage") or record.get("triage") or {}
    details = []
    if values.get("gestational_weeks") is not None:
        details.append(readback_field(lang, "weeks", weeks=values["gestational_weeks"]))
    if values.get("bp_systolic") is not None and values.get("bp_diastolic") is not None:
        details.append(readback_field(lang, "bp", sys=values["bp_systolic"], dia=values["bp_diastolic"]))
    if values.get("hb_g_dl") is not None:
        details.append(readback_field(lang, "hb", hb=values["hb_g_dl"]))
    details.append(f"screening {triage.get('level', 'unknown')}")
    action = _next_missing_action(state, require_full_checklist) or "CONFIRM_VALUES"
    prompt = t(lang, action) if action in {"ASK_PATIENT_NAME", "ASK_SPOUSE_NAME", "ASK_WEEKS", "ASK_BP", "ASK_HB", "ASK_SYMPTOMS"} else t(lang, "read_history_default_prompt")
    name = patient.get("full_name") or "patient"
    return {
        "next_action": action,
        "reply_text": t(lang, "read_history", name=name, details=", ".join(details), prompt=prompt),
    }


def history_repeat_node(state: CallState) -> dict[str, str]:
    return {
        "next_action": "ASK_READ_PRIOR_RECORD",
        "reply_text": t(_state_lang(state), "history_repeat"),
    }


def correct_node(state: CallState) -> dict[str, Any]:
    lang = _state_lang(state)
    raw_utt = state.get("utterance", "")
    roman_utt = transliterate_to_english(raw_utt).lower()
    text_for_match = f"{raw_utt} {roman_utt}"
    target: str | None = None
    reply = t(lang, "correct_generic")
    if _keyword_regex(lang, "spouse").search(text_for_match):
        target = "ASK_SPOUSE_NAME"
        reply = t(lang, "correct_spouse")
    elif _keyword_regex(lang, "patient").search(text_for_match):
        target = "ASK_PATIENT_NAME"
        reply = t(lang, "correct_patient")
    elif _keyword_regex(lang, "bp").search(text_for_match):
        target = "ASK_BP"
        reply = t(lang, "correct_bp")
    elif _keyword_regex(lang, "weeks").search(text_for_match):
        target = "ASK_WEEKS"
        reply = t(lang, "correct_weeks")
    elif _keyword_regex(lang, "hb").search(text_for_match):
        target = "ASK_HB"
        reply = t(lang, "correct_hb")
    return {
        "verification": "UNVERIFIED",
        "next_action": "CORRECT_VALUES",
        "correction_target": target,
        "reply_text": reply,
        "is_confirmation_turn": False,
    }


def merge_turn(state: CallState) -> dict[str, Any]:
    """Merge new explicit values over prior values; corrections replace old facts."""
    lang = _state_lang(state)
    current = ExtractedVisit.model_validate(state.get("visit_values", {}))
    extraction = TurnExtraction.model_validate(state.get("turn_extraction", {}))
    incoming = extraction.visit
    utterance = state.get("utterance", "")
    roman_utt = transliterate_to_english(utterance)
    text_for_match = f"{utterance} {roman_utt}"
    prev_action = state.get("correction_target") or state.get("next_action")
    is_pregnant = state.get("is_pregnant")

    no_terms = list(lexicon(lang).get("no", []))
    if lang != DEFAULT_LANGUAGE:
        no_terms.extend(lexicon(DEFAULT_LANGUAGE).get("no", []))
    no_alt = "|".join(re.escape(w) for w in dict.fromkeys(no_terms) if w)
    non_pregnant_stated = bool(
        incoming.gestational_weeks == 0
        or _NON_PREGNANT_RE.search(utterance)
        or _keyword_regex(lang, "not_pregnant").search(text_for_match)
        or (
            prev_action == "ASK_WEEKS"
            and re.search(rf"^\s*(?:0|zero|shunya|{no_alt})\b", utterance.strip(), re.I)
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
    mentions_spouse = bool(_keyword_regex(lang, "spouse").search(text_for_match))
    mentions_patient = bool(_keyword_regex(lang, "patient").search(text_for_match))
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
            changed_readback.append(readback_field(lang, "patient_name", name=current_patient["full_name"]))
        if incoming_patient.get("spouse_or_guardian_name"):
            changed_readback.append(readback_field(lang, "spouse_name", spouse=current_patient["spouse_or_guardian_name"]).lstrip(", "))
        if incoming_patient.get("village"):
            changed_readback.append(readback_field(lang, "village", village=current_patient["village"]).lstrip(", "))
        if non_pregnant_stated:
            changed_readback.append(readback_field(lang, "non_pregnant"))
        elif incoming.gestational_weeks is not None and merged_data.get("gestational_weeks") is not None:
            changed_readback.append(readback_field(lang, "weeks", weeks=merged_data["gestational_weeks"]))
        if (incoming.bp_systolic is not None or incoming.bp_diastolic is not None) and (
            merged_data.get("bp_systolic") is not None and merged_data.get("bp_diastolic") is not None
        ):
            changed_readback.append(readback_field(lang, "bp", sys=merged_data["bp_systolic"], dia=merged_data["bp_diastolic"]))
        if incoming.hb_g_dl is not None and merged_data.get("hb_g_dl") is not None:
            changed_readback.append(readback_field(lang, "hb", hb=f"{merged_data['hb_g_dl']:g}"))
        for report in incoming.symptoms:
            if report.presence is Presence.PRESENT:
                label = symptom_label(lang, report.code) or report.phrase or report.code.value.replace("_", " ").lower()
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
    lang = _state_lang(state)
    validation = ValidationResult.model_validate(state["validation_result"])
    first_error = next(issue for issue in validation.issues if issue.severity.value == "ERROR")
    prompt = t(lang, "repeat_bp") if first_error.field == "blood_pressure" else t(lang, "repeat_generic", message=first_error.message)
    triage = TriageResult.model_validate(state.get("triage_result", {"level": "ROUTINE"}))
    if triage.level in (TriageLevel.URGENT, TriageLevel.CRITICAL):
        prompt = t(lang, "repeat_escalate_prefix", level=triage.level.value, prompt=prompt)
    return {"next_action": "REPEAT_VALUE", "reply_text": prompt}


def collect_missing_node(state: CallState, *, require_full_checklist: bool) -> dict[str, str]:
    lang = _state_lang(state)
    action = _next_missing_action(state, require_full_checklist)
    if action is None:
        return {"next_action": "CONFIRM_VALUES", "reply_text": t(lang, "collect_confirm_fallback")}
    if action == "ASK_BP" and state.get("is_pregnant") is False and state.get("next_action") == "ASK_WEEKS":
        prompt = t(lang, "ASK_BP_NON_PREGNANT")
    else:
        prompt = t(lang, action)
    return {"next_action": action, "reply_text": prompt}


def readback_node(state: CallState) -> dict[str, str]:
    lang = _state_lang(state)
    visit = ExtractedVisit.model_validate(state.get("visit_values", {}))
    patient = PatientReference.model_validate(state.get("patient", {}))
    triage = TriageResult.model_validate(state["triage_result"])
    fields: list[str] = []
    if patient.full_name:
        identity = readback_field(lang, "patient_name", name=patient.full_name)
        if patient.spouse_or_guardian_name:
            identity += readback_field(lang, "spouse_name", spouse=patient.spouse_or_guardian_name)
        if patient.village:
            identity += readback_field(lang, "village", village=patient.village)
        fields.append(identity)
    if visit.gestational_weeks is not None:
        fields.append(readback_field(lang, "weeks", weeks=visit.gestational_weeks))
    elif state.get("is_pregnant") is False:
        fields.append(readback_field(lang, "non_pregnant"))
    if visit.bp_systolic is not None and visit.bp_diastolic is not None:
        fields.append(readback_field(lang, "bp", sys=visit.bp_systolic, dia=visit.bp_diastolic))
    if visit.hb_g_dl is not None:
        fields.append(readback_field(lang, "hb", hb=f"{visit.hb_g_dl:g}"))
    all_symptoms_absent = (
        len(visit.symptoms) == len(SymptomCode)
        and all(report.presence is Presence.ABSENT for report in visit.symptoms)
    )
    if all_symptoms_absent:
        fields.append(readback_field(lang, "no_danger_signs"))
    else:
        for report in visit.symptoms:
            if report.presence is Presence.PRESENT:
                label = symptom_label(lang, report.code) or report.phrase or report.code.value.replace("_", " ").lower()
                fields.append(label)
            elif report.presence is Presence.ABSENT:
                label = symptom_label(lang, report.code) or report.phrase or report.code.value.replace("_", " ").lower()
                fields.append(readback_field(lang, "symptom_absent", label=label))
    summary = ", ".join(fields) if fields else t(lang, "readback_empty")
    changed_readback = state.get("changed_readback") or []
    if changed_readback:
        changed_summary = ", ".join(changed_readback)
        reply_text = t(lang, "readback_changed", changed_summary=changed_summary, level=triage.level.value)
    else:
        reply_text = t(lang, "readback", summary=summary, level=triage.level.value)
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
            lang = _state_lang(state)
            try:
                result = await extractor(state.get("utterance", ""), effective_action, language=lang)
            except TypeError:
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
