"""Thin adapter from a completed voice transcript to the LangGraph call flow,
plus ASHA Worker ID, Non-Pregnant Visit Category, and DTMF keypad enrichment."""

from __future__ import annotations

from enum import Enum
import re
from typing import Any, Protocol

from pydantic import Field

from asha_vani.graph import build_call_graph
from asha_vani.schemas import (
    ExtractedVisit,
    PatientReference,
    Presence,
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
    asha_id: str | None = None
    asha_name: str | None = None
    visit_category: str = "MATERNAL_ANC"
    is_pregnant: bool | None = None
    english_summary: str = ""
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


_ASHA_ROSTER: dict[str, tuple[str, str]] = {
    "001": ("ASHA-001", "Sunita Kumari (ASHA-001)"),
    "002": ("ASHA-002", "Meena Yadav (ASHA-002)"),
    "003": ("ASHA-003", "Rekha Devi (ASHA-003)"),
    "004": ("ASHA-004", "Savita Sharma (ASHA-004)"),
    "005": ("ASHA-005", "Geeta Devi (ASHA-005)"),
    "006": ("ASHA-006", "Kiran Verma (ASHA-006)"),
    "007": ("ASHA-007", "Asha Kumari (ASHA-007)"),
    "008": ("ASHA-008", "Usha Devi (ASHA-008)"),
}

_NON_PREGNANT_PATTERNS = re.compile(
    r"(?:\b(?:pregnant\s+nahi|garbhvati\s+nahi|garbhini\s+nahi|not\s+pregnant|no\s+pregnancy|"
    r"postnatal|post\s+natal|pnc|delivery\s+ho\s+gayi|general\s+checkup|normal\s+checkup|bukhar\s+checkup|"
    r"0\s*(?:weeks?|hafte|hafta|saptah)|zero\s*(?:weeks?|hafte|hafta|saptah)|shunya\s*(?:weeks?|hafte|hafta|saptah))\b|"
    r"गर्भवती\s+नहीं|0\s*(?:हफ्ते|हफ़्ते|सप्ताह)|शून्य\s*(?:हफ्ते|हफ़्ते|सप्ताह))",
    re.IGNORECASE,
)


def extract_asha_worker_id(transcript: str, *, expected_action: str | None = None) -> tuple[str | None, str | None]:
    """Extract ASHA worker ID (e.g. 'ASHA-005' or bare '005' when asked for ASHA ID)."""
    norm = transcript.lower()
    for word, digit in (("zero", "0"), ("shunya", "0"), ("paanch", "5"), ("panch", "5"), ("पांच", "5"), ("पाँच", "5")):
        norm = re.sub(rf"\b{re.escape(word)}\b", digit, norm)
    norm = re.sub(r"\b(\d)\s+(\d)\s+(\d)\b", r"\1\2\3", norm)

    explicit = re.search(
        r"\b(?:asha(?:\s*id|\s*worker|\s*code|\s*number)?|id(?:\s*number)?|आशा(?:\s*आईडी)?)\s*(?:hai|is|:|-)?\s*0*(\d{1,3})\b",
        norm,
        re.IGNORECASE,
    )
    if explicit:
        code = f"{int(explicit.group(1)):03d}"
        return _ASHA_ROSTER.get(code, (f"ASHA-{code}", f"Registered Worker (ASHA-{code})"))

    if expected_action == "ASK_ASHA_ID":
        bare = re.search(r"^\s*#?\*?0*(\d{1,3})#?\s*$", norm)
        if bare:
            code = f"{int(bare.group(1)):03d}"
            return _ASHA_ROSTER.get(code, (f"ASHA-{code}", f"Registered Worker (ASHA-{code})"))
    return None, None


def detect_non_pregnant_category(transcript: str, *, expected_action: str | None = None) -> tuple[bool | None, str | None]:
    """Detect if the speaker explicitly indicates a non-pregnant / postnatal / general checkup."""
    low = transcript.lower().strip()
    is_weeks_zero_or_no = bool(
        expected_action == "ASK_WEEKS"
        and re.match(r"^\s*(?:0|zero|shunya|nahi|nahin|no)\b|^\s*(?:नहीं|शून्य)\b", low, re.IGNORECASE)
    )
    if _NON_PREGNANT_PATTERNS.search(low) or re.match(r"^\s*0\s*#?\s*$", low) or is_weeks_zero_or_no:
        if any(w in low for w in ("postnatal", "post natal", "pnc", "delivery")):
            return False, "POSTNATAL_PNC"
        if any(w in low for w in ("bukhar", "fever", "infection")):
            return False, "FEVER_INFECTION"
        return False, "GENERAL_VITALS"
    return None, None


def _normalize_dtmf_for_graph(transcript: str, expected_action: str | None) -> str:
    """Convert DTMF keypad shorthands ('150*95', '1' on confirm) into phrases that `extract_turn` understands."""
    cleaned = transcript.strip()
    bp_dtmf = re.match(r"^\s*(\d{2,3})\s*[*#]\s*(\d{2,3})#?\s*$", cleaned)
    if bp_dtmf:
        return f"BP {bp_dtmf.group(1)} by {bp_dtmf.group(2)}"
    if expected_action in ("CONFIRM_VALUES", "ASK_ANOTHER_PATIENT"):
        if re.match(r"^\s*1#?\s*$", cleaned):
            return "Haan sahi hai"
        if re.match(r"^\s*2#?\s*$", cleaned):
            return "Nahi galat hai"
    return transcript


def _build_english_summary(
    asha_id: str | None,
    visit_category: str,
    is_pregnant: bool | None,
    patient: PatientReference,
    visit: ExtractedVisit,
    triage: TriageResult,
    next_action: str,
) -> str:
    parts: list[str] = []
    if asha_id:
        parts.append(f"Worker: {asha_id}")
    if patient.full_name:
        sp = f" (Husband: {patient.spouse_or_guardian_name})" if patient.spouse_or_guardian_name else ""
        vil = f", {patient.village}" if patient.village else ""
        parts.append(f"Patient: {patient.full_name}{sp}{vil}")
    if is_pregnant is False:
        parts.append(f"Category: {visit_category.replace('_', ' ').title()} (Non-Pregnant)")
    elif visit.gestational_weeks is not None:
        parts.append(f"Gestation: {visit.gestational_weeks} wks")
    if visit.bp_systolic is not None and visit.bp_diastolic is not None:
        parts.append(f"BP: {visit.bp_systolic}/{visit.bp_diastolic} mmHg")
    if visit.hb_g_dl is not None:
        parts.append(f"Hb: {visit.hb_g_dl:g} g/dL")
    syms = [r.code.value.replace("_", " ").lower() for r in visit.symptoms if r.presence is Presence.PRESENT]
    if syms:
        parts.append("Symptoms: " + ", ".join(syms))
    rules = f" [{', '.join(triage.rules_fired)}]" if triage.rules_fired else ""
    parts.append(f"Triage: {triage.level.value}{rules} -> {next_action}")
    return " · ".join(parts)


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
        self.require_full_checklist = require_full_checklist
        self._asha_workers: dict[str, tuple[str, str]] = {}
        self._pregnancy_state: dict[str, tuple[bool | None, str]] = {}
        self._last_action: dict[str, str] = {}

    async def run_turn(self, session_id: str, utterance_seq: int, transcript: str) -> TurnOutput:
        expected_action = self._last_action.get(
            session_id,
            "ASK_ASHA_ID" if self.require_full_checklist else None,
        )

        # 1. Detect ASHA Worker ID ("ASHA ID 005" or bare "005" on initial turn)
        found_id, found_name = extract_asha_worker_id(transcript, expected_action=expected_action)
        if found_id and found_name:
            self._asha_workers[session_id] = (found_id, found_name)
        asha_id, asha_name = self._asha_workers.get(session_id, (None, None))

        # 2. Detect Non-Pregnant category ("pregnant nahi hai", "0 weeks", "postnatal", etc.)
        np_flag, np_cat = detect_non_pregnant_category(transcript, expected_action=expected_action)
        cur_preg, cur_cat = self._pregnancy_state.get(session_id, (None, "MATERNAL_ANC"))
        if np_flag is False and np_cat:
            cur_preg = False
            cur_cat = np_cat
            self._pregnancy_state[session_id] = (cur_preg, cur_cat)

        is_pure_asha_id_turn = bool(
            found_id
            and len(
                re.sub(
                    r"\b(?:mera|asha|id|number|code|hai|is|zero|shunya|paanch|panch|\d+)\b",
                    "",
                    transcript.lower(),
                ).strip(" .,!?#*")
            )
            < 3
        )

        graph_utterance = "" if is_pure_asha_id_turn else _normalize_dtmf_for_graph(transcript, expected_action)
        invoke_payload: dict[str, Any] = {"utterance": graph_utterance}
        if cur_preg is False:
            invoke_payload["is_pregnant"] = False

        state = await self._graph.ainvoke(
            invoke_payload,
            config={"configurable": {"thread_id": session_id}},
        )

        extraction = TurnExtraction.model_validate(state.get("turn_extraction", {}))
        patient = PatientReference.model_validate(state.get("patient", {}))
        visit_values = ExtractedVisit.model_validate(state.get("visit_values", {}))
        validation_result = ValidationResult.model_validate(state["validation_result"])
        triage_result = TriageResult.model_validate(state["triage_result"])
        verification = VerificationStatus(state.get("verification", "UNVERIFIED"))
        next_action = str(state.get("next_action", ""))
        reply_text = str(state.get("reply_text", ""))
        triage_summary = str(state.get("triage_summary", ""))
        is_confirmation_turn = bool(state.get("is_confirmation_turn", False))

        if state.get("is_pregnant") is False:
            cur_preg = False
            if cur_cat == "MATERNAL_ANC":
                cur_cat = "GENERAL_VITALS"
            self._pregnancy_state[session_id] = (cur_preg, cur_cat)
        elif visit_values.gestational_weeks is not None and visit_values.gestational_weeks > 0:
            cur_preg = True
            cur_cat = "MATERNAL_ANC"
            self._pregnancy_state[session_id] = (cur_preg, cur_cat)
        elif next_action == "ASK_PATIENT_NAME" and int(state.get("visit_number", 1)) > 1:
            cur_preg = None
            cur_cat = "MATERNAL_ANC"
            self._pregnancy_state[session_id] = (cur_preg, cur_cat)

        if is_pure_asha_id_turn and asha_name and asha_id and next_action == "ASK_PATIENT_NAME":
            short_name = asha_name.split("(")[0].strip()
            reply_text = f"Ji {short_name} Didi ({asha_id}), marij (patient) ka poora naam aur gaon batayein."

        self._last_action[session_id] = next_action
        eng_summary = _build_english_summary(
            asha_id=asha_id,
            visit_category=cur_cat,
            is_pregnant=cur_preg,
            patient=patient,
            visit=visit_values,
            triage=triage_result,
            next_action=next_action,
        )

        return TurnOutput(
            session_id=session_id,
            utterance_seq=utterance_seq,
            visit_number=int(state.get("visit_number", 1)),
            matched_patient_id=state.get("matched_patient_id"),
            completed_visits=list(state.get("completed_visits", [])),
            call_complete=bool(state.get("call_complete", False)),
            transcript=transcript,
            asha_id=asha_id,
            asha_name=asha_name,
            visit_category=cur_cat,
            is_pregnant=cur_preg,
            english_summary=eng_summary,
            patient=patient,
            extracted_turn=extraction.visit,
            visit_values=visit_values,
            validation_result=validation_result,
            triage_result=triage_result,
            verification=verification,
            next_action=next_action,
            reply_text=reply_text,
            triage_summary=triage_summary,
            is_confirmation_turn=is_confirmation_turn,
        )
