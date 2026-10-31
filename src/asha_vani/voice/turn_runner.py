"""TurnRunner bridge between Person A's CallSession and Person B's call_graph.

Solves the 3 integration points documented in AGENTS.md 4.1 without modifying Person B's files:
1. Extracts `ExtractedVisit` + `PatientReference` from raw Hindi/Hinglish transcript text
   (including dialect numbers like "dedh sau by pachhanve" -> 150/95).
2. Tracks `PatientReference` (`full_name`, `spouse_or_guardian_name`, `age_years`, `village`, `tola`)
   per session so the PHC Register table knows which patient the visit belongs to.
3. Handles the confirmation turn ("Haan sahi hai" / "Yes") after readback so verification flips
   from `PENDING_CONFIRMATION` (`UNVERIFIED`) to `COMMITTED_VERIFIED` (`VERIFIED`).
"""

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
    SymptomCode,
    SymptomReport,
    TriageLevel,
    TriageResult,
    ValidationResult,
)


class VerificationStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    INCOMPLETE = "INCOMPLETE"


class TurnOutput(ProjectModel):
    """Structured result returned to CallSession after one completed utterance turn."""

    session_id: str
    utterance_seq: int
    transcript: str
    patient: PatientReference
    extracted_turn: ExtractedVisit
    visit_values: ExtractedVisit
    validation_result: ValidationResult
    triage_result: TriageResult
    verification: VerificationStatus
    next_action: str
    reply_text: str
    triage_summary: str = ""
    is_confirmation_turn: bool = False


class TurnRunner(Protocol):
    """Protocol invoked by CallSession once per finished utterance."""

    async def run_turn(
        self,
        session_id: str,
        utterance_seq: int,
        transcript: str,
    ) -> TurnOutput: ...


# ---------------------------------------------------------------------------
# Hindi / Hinglish Dialect Number & Symptom Normalization
# ---------------------------------------------------------------------------
_HINDI_NUMBER_WORDS: dict[str, int] = {
    "dedh sau": 150,
    "dhedh sau": 150,
    "ek sau pachas": 150,
    "ek sau saath": 160,
    "ek sau sattar": 170,
    "ek sau assi": 180,
    "ek sau chalis": 140,
    "ek sau tees": 130,
    "ek sau bees": 120,
    "ek sau das": 110,
    "sau": 100,
    "pachhanve": 95,
    "pachanve": 95,
    "nabbe": 90,
    "pachasi": 85,
    "assi": 80,
    "pachattar": 75,
    "sattar": 70,
    "saath": 60,
    "battis": 32,
    "tees": 30,
    "atthais": 28,
    "chhabbis": 26,
    "chaubis": 24,
    "bees": 20,
}

_SYMPTOM_PATTERNS: tuple[tuple[SymptomCode, tuple[str, ...]], ...] = (
    (
        SymptomCode.SEVERE_HEADACHE,
        ("tez sar dard", "sar dard", "sir dard", "severe headache", "headache", "sar mein dard"),
    ),
    (
        SymptomCode.BLURRED_VISION,
        ("dhundhla", "blurred vision", "aankh", "dikhai nahi", "blurry"),
    ),
    (
        SymptomCode.EPIGASTRIC_PAIN,
        ("pet ke upar dard", "epigastric", "kaleje mein dard", "upper abdominal pain"),
    ),
    (
        SymptomCode.CONVULSION,
        ("daura", "jhatke", "convulsion", "seizure", "fits"),
    ),
    (
        SymptomCode.UNCONSCIOUS,
        ("behosh", "unconscious", "hosh nahi"),
    ),
    (
        SymptomCode.VAGINAL_BLEEDING,
        ("bleeding", "khoon aana", "raktstrav", "khoon beh", "spotting"),
    ),
    (
        SymptomCode.LABOUR_PAINS,
        ("prasav peeda", "labour pain", "labor pain", "dard shuru"),
    ),
    (
        SymptomCode.LEAKING_FLUID,
        ("paani nikal", "paani toot", "leaking fluid", "water broke"),
    ),
    (
        SymptomCode.REDUCED_FETAL_MOVEMENT,
        ("bacha kam ghoom", "halchal kam", "reduced movement", "fetal movement kam"),
    ),
    (
        SymptomCode.SWELLING,
        ("soojan", "sujan", "swelling", "pair phool"),
    ),
)

_CONFIRM_YES_PATTERNS = re.compile(
    r"\b(haan|han|sahi hai|bilkul sahi|theek hai|confirm|yes|correct|ok)\b",
    re.IGNORECASE,
)


def _normalize_dialect_numbers(text: str) -> str:
    """Replace common Hindi number phrases with digits before regex extraction."""
    out = text.lower()
    # Sort longest phrase first so "ek sau pachas" matches before "sau"
    for phrase, num in sorted(_HINDI_NUMBER_WORDS.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(rf"\b{re.escape(phrase)}\b", str(num), out)
    # Normalize "6 point 5" / "6 dashamlav 5" -> "6.5"
    out = re.sub(r"(\d+)\s*(?:point|dashamlav|dot)\s*(\d+)", r"\1.\2", out)
    return out


def extract_from_transcript(transcript: str) -> tuple[ExtractedVisit, PatientReference]:
    """Deterministic Hindi/Hinglish extractor for numbers, symptoms, and patient identity."""
    norm = _normalize_dialect_numbers(transcript)

    # 1. Blood pressure: e.g. "BP 150 by 95", "150/95", "BP 150 aur 95"
    bp_sys: int | None = None
    bp_dia: int | None = None
    bp_match = re.search(
        r"(?:bp|blood pressure)?\s*(\d{2,3})\s*(?:/|by|bai|over|aur|bata)\s*(\d{2,3})",
        norm,
    )
    if bp_match:
        bp_sys = int(bp_match.group(1))
        bp_dia = int(bp_match.group(2))

    # 2. Gestational weeks: e.g. "32 hafte", "32 weeks", "garbh 28 सप्ताह"
    weeks: int | None = None
    wk_match = re.search(r"\b(\d{1,2})\s*(?:hafte|hafta|week|weeks|saptah)\b", norm)
    if wk_match:
        weeks = int(wk_match.group(1))

    # 3. Hemoglobin: e.g. "hb 6.5", "hemoglobin 9.2", "khoon ki jaanch 6.5"
    hb: float | None = None
    hb_match = re.search(
        r"(?:hb|hemoglobin|haemoglobin|khoon(?:\s+ki\s+jaanch)?)\s*(?:hai|is)?\s*(\d{1,2}(?:\.\d+)?)",
        norm,
    )
    if hb_match:
        hb = float(hb_match.group(1))

    # 4. Symptoms
    symptoms: list[SymptomReport] = []
    for code, phrases in _SYMPTOM_PATTERNS:
        for phrase in phrases:
            if phrase in norm:
                # Check if negated ("nahi hai" / "no headache")
                negated = bool(
                    re.search(rf"{re.escape(phrase)}\s*(?:nahi|na)\b|\bno\s+{re.escape(phrase)}", norm)
                )
                symptoms.append(
                    SymptomReport(
                        code=code,
                        presence=Presence.ABSENT if negated else Presence.PRESENT,
                        phrase=phrase,
                    )
                )
                break

    # 5. Patient identity (Name, Spouse, Village, Tola, Age)
    patient = PatientReference()
    name_match = re.search(
        r"\b([A-Z][a-z]+\s+(?:Devi|Kumari|Bai|Begum|Yadav|Verma))\b",
        transcript,
    )
    if name_match:
        patient.full_name = name_match.group(1)

    spouse_match = re.search(
        r"(?:patni|w/o|wife of|pati)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        transcript,
        re.IGNORECASE,
    )
    if spouse_match:
        patient.spouse_or_guardian_name = spouse_match.group(1).strip()

    for v_name in ("Rampur", "Belwa", "rajpur", "sonpur"):
        if v_name.lower() in norm:
            patient.village = v_name.capitalize()
            break

    for tola_name in ("Purani Basti", "Nayi Basti", "Harijan Tola", "Dakshin Tola", "Uttar Tola"):
        if tola_name.lower() in norm:
            patient.tola = tola_name
            break

    age_match = re.search(r"\b(?:umra|age)\s*(\d{2})\b|\b(\d{2})\s*(?:saal|years)\b", norm)
    if age_match:
        patient.age_years = int(age_match.group(1) or age_match.group(2))

    visit = ExtractedVisit(
        gestational_weeks=weeks,
        bp_systolic=bp_sys,
        bp_diastolic=bp_dia,
        hb_g_dl=hb,
        symptoms=symptoms,
    )
    return visit, patient


def _has_any_clinical_value(v: ExtractedVisit) -> bool:
    return any(
        (
            v.gestational_weeks is not None,
            v.bp_systolic is not None,
            v.bp_diastolic is not None,
            v.hb_g_dl is not None,
            bool(v.symptoms),
        )
    )


class LangGraphTurnRunner:
    """Wraps Person B's `call_graph` to provide full transcript -> TurnOutput execution."""

    def __init__(self, graph: Any | None = None) -> None:
        self._graph = graph or build_call_graph()
        self._patients: dict[str, PatientReference] = {}
        self._verification: dict[str, VerificationStatus] = {}
        self._last_output: dict[str, TurnOutput] = {}

    def _merge_patient(self, session_id: str, incoming: PatientReference) -> PatientReference:
        current = self._patients.get(session_id, PatientReference())
        data = current.model_dump()
        for k, val in incoming.model_dump().items():
            if val is not None:
                data[k] = val
        merged = PatientReference.model_validate(data)
        self._patients[session_id] = merged
        return merged

    async def run_turn(
        self,
        session_id: str,
        utterance_seq: int,
        transcript: str,
    ) -> TurnOutput:
        extracted_turn, incoming_patient = extract_from_transcript(transcript)
        patient = self._merge_patient(session_id, incoming_patient)
        prev = self._last_output.get(session_id)

        # Check if this turn is a pure confirmation ("Haan sahi hai") after a readback prompt
        is_confirm_phrase = bool(_CONFIRM_YES_PATTERNS.search(transcript))
        if (
            prev is not None
            and prev.next_action == "CONFIRM_VALUES"
            and is_confirm_phrase
            and not _has_any_clinical_value(extracted_turn)
        ):
            self._verification[session_id] = VerificationStatus.VERIFIED
            patient_label = patient.full_name or "patient"
            triage = prev.triage_result
            if triage.level in (TriageLevel.CRITICAL, TriageLevel.URGENT):
                rules_str = ", ".join(triage.rules_fired)
                reply = (
                    f"Record verified for {patient_label}. "
                    f"{triage.level.value} screening ({rules_str}); "
                    "PHC has been alerted. Please refer the patient immediately."
                )
            else:
                reply = f"Record verified for {patient_label}. Routine ANC visit saved."

            out = TurnOutput(
                session_id=session_id,
                utterance_seq=utterance_seq,
                transcript=transcript,
                patient=patient,
                extracted_turn=ExtractedVisit(),
                visit_values=prev.visit_values,
                validation_result=prev.validation_result,
                triage_result=triage,
                verification=VerificationStatus.VERIFIED,
                next_action="COMMITTED_VERIFIED",
                reply_text=reply,
                triage_summary=prev.triage_summary,
                is_confirmation_turn=True,
            )
            self._last_output[session_id] = out
            return out

        # Invoke Person B's LangGraph state machine
        state_out = await self._graph.ainvoke(
            {
                "utterance": transcript,
                "extracted_turn": extracted_turn,
            },
            config={"configurable": {"thread_id": session_id}},
        )

        visit_values = ExtractedVisit.model_validate(state_out["visit_values"])
        validation_result = ValidationResult.model_validate(state_out["validation_result"])
        triage_result = TriageResult.model_validate(state_out["triage_result"])
        next_action = str(state_out["next_action"])
        reply_text = str(state_out["reply_text"])
        triage_summary = str(state_out.get("triage_summary", ""))

        # New clinical values require readback confirmation before becoming VERIFIED
        verification = VerificationStatus.UNVERIFIED
        self._verification[session_id] = verification

        out = TurnOutput(
            session_id=session_id,
            utterance_seq=utterance_seq,
            transcript=transcript,
            patient=patient,
            extracted_turn=extracted_turn,
            visit_values=visit_values,
            validation_result=validation_result,
            triage_result=triage_result,
            verification=verification,
            next_action=next_action,
            reply_text=reply_text,
            triage_summary=triage_summary,
            is_confirmation_turn=False,
        )
        self._last_output[session_id] = out
        return out

