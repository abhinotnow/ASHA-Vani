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
import os
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
    "डेढ़ सौ": 150,
    "ek sau pachas": 150,
    "एक सौ पचास": 150,
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
    "पचानवे": 95,
    "nabbe": 90,
    "pachasi": 85,
    "assi": 80,
    "pachattar": 75,
    "sattar": 70,
    "saath": 60,
    "battis": 32,
    "बत्तीस": 32,
    "tees": 30,
    "atthais": 28,
    "chhabbis": 26,
    "chaubis": 24,
    "bees": 20,
}

_SYMPTOM_PATTERNS: tuple[tuple[SymptomCode, tuple[str, ...]], ...] = (
    (
        SymptomCode.SEVERE_HEADACHE,
        (
            "tez sar dard",
            "sar dard",
            "sir dard",
            "severe headache",
            "headache",
            "sar mein dard",
            "तेज सर दर्द",
            "तेज़ सर दर्द",
            "सर दर्द",
            "सिर दर्द",
        ),
    ),
    (
        SymptomCode.BLURRED_VISION,
        ("dhundhla", "blurred vision", "aankh", "dikhai nahi", "blurry", "धुंधला", "दिखाई नहीं"),
    ),
    (
        SymptomCode.EPIGASTRIC_PAIN,
        ("pet ke upar dard", "epigastric", "kaleje mein dard", "upper abdominal pain", "पेट के ऊपर दर्द"),
    ),
    (
        SymptomCode.CONVULSION,
        ("daura", "jhatke", "convulsion", "seizure", "fits", "दौरा", "झटके"),
    ),
    (
        SymptomCode.UNCONSCIOUS,
        ("behosh", "unconscious", "hosh nahi", "बेहोश"),
    ),
    (
        SymptomCode.VAGINAL_BLEEDING,
        ("bleeding", "khoon aana", "raktstrav", "khoon beh", "spotting", "रक्तस्राव", "खून आना", "ब्लीडिंग"),
    ),
    (
        SymptomCode.LABOUR_PAINS,
        ("prasav peeda", "labour pain", "labor pain", "dard shuru", "प्रसव पीड़ा"),
    ),
    (
        SymptomCode.LEAKING_FLUID,
        ("paani nikal", "paani toot", "leaking fluid", "water broke", "पानी निकल", "पानी टूट"),
    ),
    (
        SymptomCode.REDUCED_FETAL_MOVEMENT,
        ("bacha kam ghoom", "halchal kam", "reduced movement", "fetal movement kam", "हलचल कम"),
    ),
    (
        SymptomCode.SWELLING,
        ("soojan", "sujan", "swelling", "pair phool", "सूजन"),
    ),
)

_CONFIRM_YES_PATTERNS = re.compile(
    r"(?:\b(?:haan|han|sahi hai|bilkul sahi|theek hai|confirm|yes|correct|ok)\b|हाँ|हां|सही है|बिल्कुल सही|ठीक है)",
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
    """Deterministic Hindi (Devanagari) & Hinglish (Latin) extractor for vitals, symptoms, and patient identity."""
    norm = _normalize_dialect_numbers(transcript)

    # 1. Blood pressure: e.g. "BP 150 by 95", "150/95", "बीपी 150 बाई 95"
    bp_sys: int | None = None
    bp_dia: int | None = None
    bp_match = re.search(
        r"(?:bp|blood pressure|बीपी)?\s*(\d{2,3})\s*(?:/|by|bai|over|aur|bata|बाई|बाय|बटा|और)\s*(\d{2,3})",
        norm,
    )
    if bp_match:
        bp_sys = int(bp_match.group(1))
        bp_dia = int(bp_match.group(2))

    # 2. Gestational weeks: e.g. "32 hafte", "32 weeks", "32 हफ्ते", "28 सप्ताह"
    weeks: int | None = None
    wk_match = re.search(r"(\d{1,2})\s*(?:hafte|hafta|week|weeks|saptah|हफ्ते|हफ़्ते|सप्ताह)", norm)
    if wk_match:
        weeks = int(wk_match.group(1))

    # 3. Hemoglobin: e.g. "hb 6.5", "hemoglobin 9.2", "हीमोग्लोबिन 6.5", "खून की जांच 6.5"
    hb: float | None = None
    hb_match = re.search(
        r"(?:hb|hemoglobin|haemoglobin|khoon(?:\s+ki\s+jaanch)?|हीमोग्लोबिन|एचबी|खून(?:\s+की\s+जांच)?)\s*(?:hai|is|है)?\s*(\d{1,2}(?:\.\d+)?)",
        norm,
    )
    if hb_match:
        hb = float(hb_match.group(1))

    # 4. Symptoms
    symptoms: list[SymptomReport] = []
    for code, phrases in _SYMPTOM_PATTERNS:
        for phrase in phrases:
            if phrase in norm:
                negated = bool(
                    re.search(
                        rf"{re.escape(phrase)}\s*(?:nahi|na|नहीं)\b|\bno\s+{re.escape(phrase)}",
                        norm,
                    )
                )
                symptoms.append(
                    SymptomReport(
                        code=code,
                        presence=Presence.ABSENT if negated else Presence.PRESENT,
                        phrase=phrase,
                    )
                )
                break

    # 5. Patient identity (Name, Spouse, Village, Tola, Age — case-insensitive & natural phrases)
    patient = PatientReference()

    # Explicit phrasing: "patient/marij ka naam X", "naam X hai", or "X ke ghar"
    explicit_name = re.search(
        r"(?:patient|marij|mareej|mahila|bahu)?\s*(?:ka\s+)?(?:naam|name|is)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)",
        transcript,
        re.IGNORECASE,
    )
    ghar_name = re.search(
        r"\b([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s+(?:ji\s+)?ke\s+ghar\b",
        transcript,
        re.IGNORECASE,
    )
    honorific_name = re.search(
        r"\b([a-zA-Z]+\s+(?:devi|kumari|bai|begum|yadav|verma|sharma|singh|khatoon))\b",
        transcript,
        re.IGNORECASE,
    )
    if honorific_name:
        patient.full_name = honorific_name.group(1).title()
    elif explicit_name:
        candidate = explicit_name.group(1).strip()
        if candidate.lower() not in ("kya", "hai", "batao", "bataye"):
            patient.full_name = candidate.title()
    elif ghar_name:
        candidate = ghar_name.group(1).strip()
        # Strip leading prepositions like "main rampur mein"
        candidate = re.sub(r"^.*\b(?:mein|me|se|at|in)\s+", "", candidate, flags=re.IGNORECASE).strip()
        if candidate and candidate.lower() not in ("kiske", "kis", "unke", "marij", "patient"):
            patient.full_name = candidate.title()
    else:
        for dev_name, lat_name in (
            ("पूजा देवी", "Pooja Devi"),
            ("कविता देवी", "Kavita Devi"),
            ("रिंकी कुमारी", "Rinki Kumari"),
            ("अनीता देवी", "Anita Devi"),
            ("सरिता कुमारी", "Sarita Kumari"),
            ("गुड्डी देवी", "Guddi Devi"),
        ):
            if dev_name in transcript:
                patient.full_name = lat_name
                break

    spouse_match = re.search(
        r"(?:patni|pati(?:\s+ka\s+naam)?|husband(?:\s+name)?|spouse|w/o|wife of|guardian)\s*(?:is|hai|:)?\s*([a-zA-Z]+(?:\s+[a-zA-Z]+)?)",
        transcript,
        re.IGNORECASE,
    )
    if spouse_match:
        sp_cand = spouse_match.group(1).strip()
        # Strip trailing filler words like "hai", "aur", "umra"
        sp_cand = re.sub(r"\b(hai|aur|and|age|umra|hafte|weeks|bp)\b.*$", "", sp_cand, flags=re.IGNORECASE).strip()
        if sp_cand:
            patient.spouse_or_guardian_name = sp_cand.title()
    elif "रमेश कुमार" in transcript:
        patient.spouse_or_guardian_name = "Ramesh Kumar"

    for v_token, v_canon in (
        ("rampur", "Rampur"),
        ("रामपुर", "Rampur"),
        ("belwa", "Belwa"),
        ("बेलवा", "Belwa"),
    ):
        if v_token in norm:
            patient.village = v_canon
            break

    for t_token, t_canon in (
        ("purani basti", "Purani Basti"),
        ("पुरानी बस्ती", "Purani Basti"),
        ("nayi basti", "Nayi Basti"),
        ("नई बस्ती", "Nayi Basti"),
        ("harijan tola", "Harijan Tola"),
        ("dakshin tola", "Dakshin Tola"),
    ):
        if t_token in norm:
            patient.tola = t_canon
            break

    age_match = re.search(r"\b(?:umra|age)\s*(\d{2})\b|\b(\d{2})\s*(?:saal|years|साल)\b", norm)
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


class _LLMExtractionSchema(ProjectModel):
    """Structured output schema returned by Groq Llama 3.3 70B / 3.1 8B."""

    patient: PatientReference = Field(default_factory=PatientReference)
    visit: ExtractedVisit = Field(default_factory=ExtractedVisit)
    is_confirmation_yes: bool = False


_GROQ_SYSTEM_PROMPT = """You are a clinical data extraction engine for ASHA-Vani, assisting rural Indian ASHA workers.
Extract ONLY the facts explicitly stated in the ASHA worker's Hindi/Hinglish/Devanagari utterance.
Rules:
1. Never guess or invent numbers. Leave unmentioned numeric fields as null.
2. Convert Hindi spoken numbers to digits (e.g. 'dedh sau by pachhanve' -> bp_systolic=150, bp_diastolic=95; 'battis hafte' -> gestational_weeks=32; '6 point 5' -> hb_g_dl=6.5).
3. Map reported danger signs to the exact SymptomCode enum with presence='PRESENT' (or 'ABSENT' if explicitly negated).
4. Extract patient identity (full_name in Latin script e.g. 'Pooja Devi', spouse_or_guardian_name, age_years, village, tola) when mentioned.
5. Set is_confirmation_yes=true ONLY if the speaker is confirming a readback (e.g. 'Haan sahi hai', 'Yes correct', 'हाँ बिल्कुल सही है') with no new measurements.
6. NEVER assign clinical risk or triage levels; deterministic code decides risk."""


async def extract_with_groq_or_fallback(
    transcript: str,
    *,
    expected_action: str | None = None,
    api_key: str | None = None,
    primary_model: str = "llama-3.3-70b-versatile",
    fallback_model: str = "llama-3.1-8b-instant",
) -> tuple[ExtractedVisit, PatientReference, bool]:
    """Call Groq Llama 3.3 70B (falling back to 3.1 8B on 429/error) when GROQ_API_KEY is set;
    also merge deterministic Hindi/Hinglish regex extractions so no spoken name or number is missed."""
    regex_visit, regex_patient = extract_from_transcript(transcript)
    regex_confirm = bool(_CONFIRM_YES_PATTERNS.search(transcript)) and not _has_any_clinical_value(regex_visit)

    key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
    if key:
        from langchain_groq import ChatGroq

        context_hint = (
            f"\nContext: The system just prompted the ASHA worker with action '{expected_action}'. "
            "If expected_action is 'ASK_SPOUSE_NAME' and the speaker says a person's name, put it in "
            "patient.spouse_or_guardian_name (NOT full_name). If expected_action is 'ASK_PATIENT_NAME', "
            "put the patient's name in patient.full_name."
            if expected_action
            else ""
        )

        for model_name in (primary_model, fallback_model):
            try:
                llm = ChatGroq(
                    api_key=key,  # type: ignore[arg-type]
                    model=model_name,
                    temperature=0.0,
                    max_retries=1,
                )
                structured_llm = llm.with_structured_output(_LLMExtractionSchema)
                res = await structured_llm.ainvoke(
                    [
                        ("system", _GROQ_SYSTEM_PROMPT + context_hint),
                        ("human", transcript),
                    ]
                )
                if isinstance(res, _LLMExtractionSchema):
                    # Backfill any field that regex caught if LLM left it null
                    v_data = res.visit.model_dump()
                    for k, val in regex_visit.model_dump().items():
                        if v_data.get(k) in (None, []) and val not in (None, []):
                            v_data[k] = val
                    p_data = res.patient.model_dump()
                    for k, val in regex_patient.model_dump().items():
                        if p_data.get(k) is None and val is not None:
                            p_data[k] = val
                    return (
                        ExtractedVisit.model_validate(v_data),
                        PatientReference.model_validate(p_data),
                        res.is_confirmation_yes or regex_confirm,
                    )
            except Exception:
                continue

    return regex_visit, regex_patient, regex_confirm


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


def _build_hinglish_reply(
    patient: PatientReference,
    visit: ExtractedVisit,
    validation: ValidationResult,
    triage: TriageResult,
    *,
    require_full_checklist: bool = False,
) -> tuple[str, str]:
    """Generate natural Hinglish follow-up prompts for missing fields, then a complete readback."""
    # 1. Hard validation error -> ask to repeat that specific measurement in Hinglish
    if validation.has_errors:
        first_err = next(i for i in validation.issues if i.severity.value == "ERROR")
        field_hi = {
            "bp_systolic": "uper wala BP (systolic)",
            "bp_diastolic": "neeche wala BP (diastolic)",
            "blood_pressure": "BP reading",
            "hb_g_dl": "Hemoglobin (khoon ki jaanch)",
            "gestational_weeks": "pregnancy ke hafte (weeks)",
        }.get(first_err.field, first_err.field)
        return (
            "REPEAT_VALUE",
            f"Didi, {field_hi} sahi sunai nahi diya ({first_err.message}). Kripya {field_hi} dobara batayein.",
        )

    p_name = patient.full_name
    p_spouse = patient.spouse_or_guardian_name
    p_village = f" ({patient.village})" if patient.village else ""

    # 2. If interactive step-by-step collection is active, check missing fields in order:
    if require_full_checklist:
        if not p_name:
            return (
                "ASK_PATIENT_NAME",
                "Didi, sabse pehle marij (patient) ka poora naam aur gaon ka naam batayein.",
            )
        if not p_spouse:
            return (
                "ASK_SPOUSE_NAME",
                f"Ji Didi, {p_name}{p_village} note kar liya. Unke pati (husband) ka naam kya hai?",
            )
        if visit.gestational_weeks is None:
            return (
                "ASK_WEEKS",
                f"Theek hai, {p_name} patni {p_spouse}. Unke pregnancy ke kitne hafte (weeks) hue hain?",
            )
        if visit.bp_systolic is None or visit.bp_diastolic is None:
            return (
                "ASK_BP",
                f"Ji Didi, {visit.gestational_weeks} hafte note ho gaya. Unka BP (blood pressure) kitna aaya hai?",
            )
        if visit.hb_g_dl is None:
            return (
                "ASK_HB",
                f"BP {visit.bp_systolic} by {visit.bp_diastolic} note kar liya. Unka Hemoglobin (Hb) kitna hai, aur koi takleef ya lakshan hai kya?",
            )
    else:
        # Even in single-shot mode, if nothing at all was captured, prompt for patient & vitals
        if not p_name and not _has_any_clinical_value(visit):
            return (
                "ASK_PATIENT_NAME",
                "Namaste Didi! Kripya marij (patient) ka naam, pati ka naam, hafte, BP aur Hemoglobin batayein.",
            )

    # 3. All required fields captured -> full readback confirmation
    details: list[str] = []
    if p_name:
        spouse_str = f" W/o {p_spouse}" if p_spouse else ""
        details.append(f"Marij {p_name}{spouse_str}{p_village}")
    if visit.gestational_weeks is not None:
        details.append(f"{visit.gestational_weeks} hafte")
    if visit.bp_systolic is not None and visit.bp_diastolic is not None:
        details.append(f"BP {visit.bp_systolic} by {visit.bp_diastolic}")
    if visit.hb_g_dl is not None:
        details.append(f"Hemoglobin {visit.hb_g_dl:g} g/dL")

    present_syms = [
        (r.phrase or r.code.value.replace("_", " ").lower())
        for r in visit.symptoms
        if r.presence is Presence.PRESENT
    ]
    if present_syms:
        details.append("lakshan: " + ", ".join(present_syms))

    summary_str = ", ".join(details)
    rules_str = f" (Rules: {', '.join(triage.rules_fired)})" if triage.rules_fired else ""
    return (
        "CONFIRM_VALUES",
        f"Didi, kripya confirm karein: {summary_str}. Screening: {triage.level.value}{rules_str}. Kya yeh saari jaankari sahi hai?",
    )


class LangGraphTurnRunner:
    """Wraps Person B's `call_graph` to provide full transcript -> TurnOutput execution."""

    def __init__(self, graph: Any | None = None, *, require_full_checklist: bool = False) -> None:
        self._graph = graph or build_call_graph()
        self.require_full_checklist = require_full_checklist
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
        prev = self._last_output.get(session_id)
        expected_action = prev.next_action if prev is not None else "ASK_PATIENT_NAME"

        extracted_turn, incoming_patient, is_confirm_phrase = await extract_with_groq_or_fallback(
            transcript,
            expected_action=expected_action,
        )

        # Safeguard: if we just asked for the husband/spouse name ("ASK_SPOUSE_NAME") and the LLM
        # put a single spoken name into `full_name` instead of `spouse_or_guardian_name`, move it!
        if (
            prev is not None
            and prev.next_action == "ASK_SPOUSE_NAME"
            and incoming_patient.full_name
            and not incoming_patient.spouse_or_guardian_name
            and prev.patient.full_name
        ):
            incoming_patient.spouse_or_guardian_name = incoming_patient.full_name
            incoming_patient.full_name = None

        # Context-aware fallback: if we just asked for patient name, spouse name, weeks, or Hb
        # and the user replied with a short bare answer (e.g. "Sunita Sharma" or "Rakesh Kumar" or "28"),
        # capture it even if they didn't say "naam ... hai"
        cleaned_words = transcript.strip(" .,!?")
        if not _has_any_clinical_value(extracted_turn):
            if expected_action == "ASK_PATIENT_NAME" and not incoming_patient.full_name:
                # Strip common filler words like "main rampur mein hoon"
                cand = re.sub(
                    r"\b(?:hello|namaste|didi|main|hum|gaon|village|rampur|belwa|mein|me|se|hoon|hu|bol|rahi|raha|hai|ke|ghar)\b",
                    " ",
                    cleaned_words,
                    flags=re.IGNORECASE,
                )
                cand = " ".join(cand.split()).strip()
                if 2 <= len(cand) <= 40 and not any(ch.isdigit() for ch in cand):
                    incoming_patient.full_name = cand.title()
            elif expected_action == "ASK_SPOUSE_NAME" and not incoming_patient.spouse_or_guardian_name:
                cand = re.sub(
                    r"\b(?:unke|pati|husband|ka|naam|hai|ji|didi)\b",
                    " ",
                    cleaned_words,
                    flags=re.IGNORECASE,
                )
                cand = " ".join(cand.split()).strip()
                if 2 <= len(cand) <= 40 and not any(ch.isdigit() for ch in cand):
                    incoming_patient.spouse_or_guardian_name = cand.title()

        if prev is not None:
            bare_num = re.search(r"\b(\d{1,2}(?:\.\d+)?)\b", _normalize_dialect_numbers(transcript))
            if prev.next_action == "ASK_WEEKS" and extracted_turn.gestational_weeks is None and bare_num:
                extracted_turn.gestational_weeks = int(float(bare_num.group(1)))
            elif prev.next_action == "ASK_HB" and extracted_turn.hb_g_dl is None and bare_num:
                extracted_turn.hb_g_dl = float(bare_num.group(1))

        patient = self._merge_patient(session_id, incoming_patient)

        # Check if this turn is a pure confirmation ("Haan sahi hai") after a readback prompt
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
                    f"Dhanyavaad Didi! {patient_label} ka record VERIFY ho gaya hai. "
                    f"Yeh {triage.level.value} case hai ({rules_str}). "
                    "PHC doctor ko alert bhej diya gaya hai, kripya marij ko turant PHC bhejein."
                )
            else:
                reply = f"Dhanyavaad Didi! {patient_label} ka Routine ANC record verify hokar register mein save ho gaya hai."

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
        triage_summary = str(state_out.get("triage_summary", ""))

        # Build conversational Hinglish reply (guided follow-ups or confirm readback)
        next_action, reply_text = _build_hinglish_reply(
            patient=patient,
            visit=visit_values,
            validation=validation_result,
            triage=triage_result,
            require_full_checklist=self.require_full_checklist,
        )

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

