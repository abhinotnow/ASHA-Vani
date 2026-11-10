"""Transcript-to-facts extraction. This module never assigns clinical risk."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from pydantic import Field

from asha_vani.schemas import (
    ConfirmationIntent,
    ExtractedVisit,
    PatientReference,
    Presence,
    ProjectModel,
    SymptomCode,
    SymptomReport,
    TurnExtraction,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Extract only facts explicitly stated in the ASHA worker's utterance.
The utterance may be Hindi, Hinglish, Devanagari, or code-switched.
Never infer missing measurements, diagnoses, risk levels, or medical advice.
Return patient details and visit values using the supplied schema.
Normalize spoken numbers only when clear (for example, 'dedh sau by pachhanave'
means BP 150/95). Extract gestational weeks only when weeks are stated; do not
convert months. Map only symptoms in the schema, marking PRESENT or ABSENT only
when explicitly reported/denied; use UNKNOWN when mentioned but unclear.
Set confirmation_intent to YES or NO only when answering a previous readback.
If the previous action asked for a bare answer, interpret it in that context.
The model extracts facts only; deterministic Python rules assign triage."""

_NUMBER_PHRASES: tuple[tuple[str, str], ...] = (
    ("dedh sau", "150"), ("dhedh sau", "150"), ("डेढ़ सौ", "150"),
    ("ek sau pachas", "150"), ("एक सौ पचास", "150"),
    ("ek sau saath", "160"), ("ek sau sattar", "170"),
    ("ek sau assi", "180"), ("ek sau chalis", "140"),
    ("ek sau tees", "130"), ("ek sau bees", "120"),
    ("ek sau das", "110"), ("pachhanave", "95"),
    ("pachhanve", "95"), ("pachanve", "95"), ("पचानवे", "95"),
    ("nabbe", "90"), ("pachasi", "85"), ("assi", "80"),
    ("pachattar", "75"), ("sattar", "70"), ("saath", "60"),
    ("battis", "32"), ("बत्तीस", "32"), ("atthais", "28"),
    ("chhabbis", "26"), ("chaubis", "24"), ("tees", "30"),
    ("bees", "20"), ("sau", "100"),
)

_SYMPTOM_TERMS: dict[SymptomCode, tuple[str, ...]] = {
    SymptomCode.SEVERE_HEADACHE: ("severe headache", "headache", "sar dard", "sir dard", "तेज सर दर्द", "सर दर्द", "सिर दर्द"),
    SymptomCode.BLURRED_VISION: ("blurred vision", "dhundhla", "दिखाई धुंधला", "धुंधला दिख"),
    SymptomCode.EPIGASTRIC_PAIN: ("epigastric", "upper abdominal pain", "pet ke upar dard", "पेट के ऊपर दर्द"),
    SymptomCode.CONVULSION: ("convulsion", "seizure", "fits", "daura", "jhatke", "दौरा", "झटके"),
    SymptomCode.UNCONSCIOUS: ("unconscious", "behosh", "बेहोश"),
    SymptomCode.VAGINAL_BLEEDING: ("vaginal bleeding", "bleeding", "raktstrav", "खून आना", "रक्तस्राव", "ब्लीडिंग"),
    SymptomCode.LABOUR_PAINS: ("labour pain", "labor pain", "prasav peeda", "प्रसव पीड़ा"),
    SymptomCode.LEAKING_FLUID: ("leaking fluid", "water broke", "paani nikal", "पानी निकल", "पानी टूट"),
    SymptomCode.REDUCED_FETAL_MOVEMENT: ("reduced fetal movement", "fetal movement kam", "halchal kam", "हलचल कम"),
    SymptomCode.SWELLING: ("swelling", "soojan", "sujan", "सूजन"),
}

_YES_RE = re.compile(r"\b(haan|han|yes|correct|bilkul sahi|sahi hai|theek hai)\b|हाँ|हां|बिल्कुल सही|सही है|ठीक है", re.I)
_NO_RE = re.compile(r"\b(no|nahi|nahin|galat|wrong|incorrect)\b|नहीं|गलत", re.I)


class _ModelOutput(ProjectModel):
    patient: PatientReference = Field(default_factory=PatientReference)
    visit: ExtractedVisit = Field(default_factory=ExtractedVisit)
    confirmation_intent: ConfirmationIntent = ConfirmationIntent.UNCLEAR


def _normalize_numbers(text: str) -> str:
    result = text.lower()
    for phrase, digits in sorted(_NUMBER_PHRASES, key=lambda pair: -len(pair[0])):
        result = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", digits, result)
    return re.sub(r"(\d+)\s*(?:point|dot|dashamlav)\s*(\d+)", r"\1.\2", result)


def _fallback_extract(transcript: str, expected_action: str | None) -> TurnExtraction:
    """Conservative local extraction for outages and development without API keys."""
    normalized = _normalize_numbers(transcript)
    visit_data: dict[str, Any] = {}
    patient = PatientReference()

    bp = re.search(r"(?:bp|blood pressure|बीपी)?\s*(\d{2,3})\s*(?:/|by|bai|over|बाई|बाय|और)\s*(\d{2,3})", normalized)
    if bp:
        visit_data.update(bp_systolic=int(bp.group(1)), bp_diastolic=int(bp.group(2)))
    weeks = re.search(r"(\d{1,2})\s*(?:weeks?|hafte|hafta|saptah|हफ्ते|हफ़्ते|सप्ताह)", normalized)
    if weeks:
        visit_data["gestational_weeks"] = int(weeks.group(1))
    hb = re.search(r"(?:\bhb\b|hemoglobin|haemoglobin|हीमोग्लोबिन|एचबी)\s*(?:hai|is|है)?\s*(\d{1,2}(?:\.\d+)?)", normalized)
    if hb:
        visit_data["hb_g_dl"] = float(hb.group(1))

    symptoms: list[SymptomReport] = []
    for code, terms in _SYMPTOM_TERMS.items():
        found = next((term for term in terms if term in normalized), None)
        if found:
            start = normalized.find(found)
            prefix = normalized[max(0, start - 18):start]
            suffix = normalized[start + len(found):start + len(found) + 20]
            negated = bool(
                re.search(r"(?:nahi|nahin|no|not|नहीं)\s*$", prefix)
                or re.match(r"\s*(?:nahi|nahin|no|not|नहीं)(?:\s|$)", suffix)
            )
            symptoms.append(SymptomReport(code=code, presence=Presence.ABSENT if negated else Presence.PRESENT, phrase=found))
    # A clear general denial records known symptoms as absent, never as present.
    if not symptoms and re.search(r"\b(no symptoms|koi takleef nahi|koi lakshan nahi)\b|कोई तकलीफ नहीं|कोई लक्षण नहीं", normalized):
        symptoms = [SymptomReport(code=code, presence=Presence.ABSENT) for code in SymptomCode]
    if symptoms:
        visit_data["symptoms"] = symptoms

    name = re.search(r"\b([A-Za-z]+\s+(?:devi|kumari|bai|begum|yadav|verma|sharma|singh|khatoon))\b", transcript, re.I)
    if name:
        patient.full_name = name.group(1).title()
    else:
        for devanagari_name, latin_name in (
            ("पूजा देवी", "Pooja Devi"), ("कविता देवी", "Kavita Devi"),
            ("रिंकी कुमारी", "Rinki Kumari"), ("अनीता देवी", "Anita Devi"),
            ("सरिता कुमारी", "Sarita Kumari"), ("गुड्डी देवी", "Guddi Devi"),
        ):
            if devanagari_name in transcript:
                patient.full_name = latin_name
                break
    spouse = re.search(r"(?:patni|pati(?: ka naam)?|husband(?: name)?|spouse|w/o|wife of|guardian)\s*(?:is|hai|:)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)?)", transcript, re.I)
    if spouse:
        candidate = re.sub(r"\b(hai|aur|age|weeks?|hafte|bp)\b.*$", "", spouse.group(1), flags=re.I).strip()
        if candidate:
            patient.spouse_or_guardian_name = candidate.title()
    elif "रमेश कुमार" in transcript:
        patient.spouse_or_guardian_name = "Ramesh Kumar"
    for token, canonical in (("rampur", "Rampur"), ("रामपुर", "Rampur"), ("belwa", "Belwa"), ("बेलवा", "Belwa")):
        if token in normalized:
            patient.village = canonical
            break
    for token, canonical in (
        ("purani basti", "Purani Basti"), ("पुरानी बस्ती", "Purani Basti"),
        ("nayi basti", "Nayi Basti"), ("नई बस्ती", "Nayi Basti"),
        ("harijan tola", "Harijan Tola"), ("dakshin tola", "Dakshin Tola"),
    ):
        if token in normalized:
            patient.tola = canonical
            break
    age = re.search(r"\b(?:age|umra)\s*(\d{2})\b|\b(\d{2})\s*(?:saal|years|साल)\b", normalized)
    if age:
        patient.age_years = int(age.group(1) or age.group(2))

    if expected_action == "ASK_PATIENT_NAME" and patient.full_name is None and not visit_data:
        candidate = re.sub(r"\b(?:didi|naam|hai|patient|marij|ji)\b", " ", transcript, flags=re.I)
        candidate = " ".join(candidate.split()).strip(" ,.!?")
        if 2 <= len(candidate) <= 40 and not any(char.isdigit() for char in candidate):
            patient.full_name = candidate.title()
    if expected_action == "ASK_SPOUSE_NAME" and patient.spouse_or_guardian_name is None and not visit_data:
        candidate = re.sub(r"\b(?:pati|husband|naam|hai|ji|didi|unke)\b", " ", transcript, flags=re.I)
        candidate = " ".join(candidate.split()).strip(" ,.!?")
        if 2 <= len(candidate) <= 40 and not any(char.isdigit() for char in candidate):
            patient.spouse_or_guardian_name = candidate.title()
    if expected_action == "ASK_WEEKS" and "gestational_weeks" not in visit_data:
        bare = re.search(r"\b(\d{1,2})\b", normalized)
        if bare:
            visit_data["gestational_weeks"] = int(bare.group(1))
    if expected_action == "ASK_HB" and "hb_g_dl" not in visit_data:
        bare = re.search(r"\b(\d{1,2}(?:\.\d+)?)\b", normalized)
        if bare:
            visit_data["hb_g_dl"] = float(bare.group(1))

    has_values = bool(visit_data) or any(patient.model_dump().values())
    confirmation = ConfirmationIntent.UNCLEAR
    if expected_action == "CONFIRM_VALUES" and not has_values:
        if _YES_RE.search(transcript):
            confirmation = ConfirmationIntent.YES
        elif _NO_RE.search(transcript):
            confirmation = ConfirmationIntent.NO
    return TurnExtraction(patient=patient, visit=ExtractedVisit.model_validate(visit_data), confirmation_intent=confirmation)


async def extract_turn(
    transcript: str,
    expected_action: str | None = None,
    *,
    llm: Any | None = None,
) -> TurnExtraction:
    """Extract one utterance asynchronously; fall back safely if Groq is unavailable."""
    fallback = _fallback_extract(transcript, expected_action)
    models: list[Any] = []
    if llm is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if api_key:
            try:
                from langchain_groq import ChatGroq
            except Exception as exc:
                logger.warning("Could not load Groq integration; using fallback: %s", exc)
                return fallback
            model_names = (
                os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant"),
            )
            for model_name in dict.fromkeys(model_names):
                try:
                    models.append(
                        ChatGroq(
                            api_key=api_key,
                            model=model_name,
                            temperature=0,
                            max_retries=1,
                        ).with_structured_output(_ModelOutput)
                    )
                except Exception as exc:
                    logger.warning("Could not initialize Groq model %s: %s", model_name, exc)
        else:
            return fallback
    else:
        models = [llm]

    context = f"Previous prompt/action: {expected_action or 'none'}. Interpret a short answer using this context."
    messages = [("system", SYSTEM_PROMPT + "\n" + context), ("human", transcript)]
    for model in models:
        try:
            result = await model.ainvoke(messages)
            parsed = result if isinstance(result, _ModelOutput) else _ModelOutput.model_validate(result)
            return TurnExtraction(
                patient=_merge_patient(parsed.patient, fallback.patient),
                visit=_merge_visit(parsed.visit, fallback.visit),
                confirmation_intent=(
                    parsed.confirmation_intent
                    if parsed.confirmation_intent is not ConfirmationIntent.UNCLEAR
                    else fallback.confirmation_intent
                ),
            )
        except Exception as exc:
            logger.warning("Structured extraction attempt failed; trying next provider/fallback: %s", exc)
    return fallback


def _merge_patient(primary: PatientReference, fallback: PatientReference) -> PatientReference:
    values = primary.model_dump()
    for key, value in fallback.model_dump().items():
        if values.get(key) is None and value is not None:
            values[key] = value
    return PatientReference.model_validate(values)


def _merge_visit(primary: ExtractedVisit, fallback: ExtractedVisit) -> ExtractedVisit:
    values = primary.model_dump()
    fallback_values = fallback.model_dump()
    for key, value in fallback_values.items():
        if values.get(key) in (None, []) and value not in (None, []):
            values[key] = value
    return ExtractedVisit.model_validate(values)
