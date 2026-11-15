"""Transcript-to-facts extraction. This module never assigns clinical risk."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
import unicodedata

import httpx
from pydantic import Field

from asha_vani.language.locale import DEFAULT_LANGUAGE, lexicon, locale_meta, normalize_language_code
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
IMPORTANT: Always transliterate patient.full_name, patient.spouse_or_guardian_name,
patient.village, and patient.tola into English Roman script in Title Case
(for example, 'सुनीता देवी' -> 'Sunita Devi', 'अभिनव प्रणव राज' -> 'Abhinav Pranav Raj',
'रामपुर' -> 'Rampur'). Never return Devanagari script in patient fields.
Normalize spoken numbers only when clear (for example, 'dedh sau by pachhanave'
means BP 150/95). Extract gestational weeks only when positive weeks are stated;
if the speaker says 0 weeks, 0 hafte, 0, or that the patient is not pregnant,
leave gestational_weeks as null. Do not convert months.
When the previous prompt/action is ASK_SPOUSE_NAME, or when the speaker is stating
or correcting the husband's name ('pati ka naam ...'), put the name ONLY in
patient.spouse_or_guardian_name and leave patient.full_name as null.
Map symptoms in the schema, including stomach/upper abdominal pain ('pet dard',
'pet mein dard', 'pet ke upar dard', 'stomach pain') as EPIGASTRIC_PAIN, marking
PRESENT or ABSENT only when explicitly reported/denied; use UNKNOWN when unclear.
Set confirmation_intent to YES or NO when answering a previous readback or a
yes/no question such as whether another patient should be recorded. If the
previous action asked for a bare answer, interpret it in that context.
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
    ("shunya", "0"), ("शून्य", "0"), ("zero", "0"),
)

_SYMPTOM_TERMS: dict[SymptomCode, tuple[str, ...]] = {
    SymptomCode.SEVERE_HEADACHE: (
        "severe headache", "headache", "tez sar dard", "tez sir dard",
        "sar dard", "sir dard", "sar mein dard", "sir mein dard",
        "तेज सर दर्द", "तेज सिर दर्द", "सर दर्द", "सिर दर्द", "सिर में दर्द",
    ),
    SymptomCode.BLURRED_VISION: (
        "blurred vision", "blurry vision", "dhundhla", "dhundla",
        "ankhon ke aage andhera", "दिखाई धुंधला", "धुंधला दिख", "धुंधला",
    ),
    SymptomCode.EPIGASTRIC_PAIN: (
        "epigastric", "upper abdominal pain", "abdominal pain",
        "stomach pain", "stomach ache", "stomachache",
        "pet ke upar dard", "pet mein dard", "pet me dard", "pet dard",
        "kaleje mein dard", "पेट के ऊपर दर्द", "पेट में दर्द", "पेट दर्द",
    ),
    SymptomCode.CONVULSION: (
        "convulsion", "seizure", "fits", "daura", "jhatke", "chamki",
        "दौरा", "झटके", "चमकी",
    ),
    SymptomCode.UNCONSCIOUS: (
        "unconscious", "behosh", "hosh nahi", "बेहोश", "होश नहीं",
    ),
    SymptomCode.VAGINAL_BLEEDING: (
        "vaginal bleeding", "bleeding", "heavy bleeding", "raktstrav",
        "khoon aana", "khoon behna", "खून आना", "रक्तस्राव", "ब्लीडिंग",
    ),
    SymptomCode.LABOUR_PAINS: (
        "labour pain", "labor pain", "prasav peeda", "prasav dard",
        "dard shuru", "प्रसव पीड़ा", "प्रसव दर्द",
    ),
    SymptomCode.LEAKING_FLUID: (
        "leaking fluid", "water broke", "paani nikal", "paani aana",
        "paani toot", "पानी निकल", "पानी आना", "पानी टूट",
    ),
    SymptomCode.REDUCED_FETAL_MOVEMENT: (
        "reduced fetal movement", "fetal movement kam", "halchal kam",
        "bachche ki halchal kam", "kicks kam", "हलचल कम", "बच्चे की हलचल कम",
    ),
    SymptomCode.SWELLING: (
        "swelling", "edema", "oedema", "soojan", "sujan",
        "haath pair mein soojan", "muh par soojan", "सूजन",
    ),
}

_YES_RE = re.compile(r"\b(haan|han|yes|correct|bilkul sahi|sahi hai|theek hai)\b|हाँ|हां|बिल्कुल सही|सही है|ठीक है", re.I)
_NO_RE = re.compile(r"\b(no|nahi|nahin|galat|wrong|incorrect)\b|नहीं|गलत", re.I)

_DEVANAGARI_WORD_MAP: dict[str, str] = {
    "पूजा": "Pooja", "सुनीता": "Sunita", "कविता": "Kavita", "अनीता": "Anita",
    "सरिता": "Sarita", "गुड्डी": "Guddi", "रिंकी": "Rinki", "मीना": "Meena",
    "रेखा": "Rekha", "गीता": "Geeta", "सीता": "Sita", "रीता": "Reeta",
    "राधा": "Radha", "सविता": "Savita", "प्रिया": "Priya", "नेहा": "Neha",
    "आशा": "Asha", "उषा": "Usha", "किरण": "Kiran", "लता": "Lata",
    "अभिनव": "Abhinav", "प्रणव": "Pranav", "राज": "Raj", "रमेश": "Ramesh",
    "राकेश": "Rakesh", "सुरेश": "Suresh", "महेश": "Mahesh", "मुकेश": "Mukesh",
    "संजय": "Sanjay", "विजय": "Vijay", "अजय": "Ajay", "मनोज": "Manoj",
    "पप्पू": "Pappu", "राजू": "Raju", "मोहन": "Mohan", "सोहन": "Sohan",
    "रोहन": "Rohan", "अमित": "Amit", "सुमित": "Sumit", "रोहित": "Rohit",
    "मोहित": "Mohit", "राहुल": "Rahul", "विकास": "Vikas", "दीपक": "Deepak",
    "अशोक": "Ashok", "विनोद": "Vinod", "प्रमोद": "Pramod",
    "देवी": "Devi", "कुमारी": "Kumari", "कुमार": "Kumar", "यादव": "Yadav",
    "शर्मा": "Sharma", "वर्मा": "Verma", "सिंह": "Singh", "महतो": "Mahto",
    "लाल": "Lal", "पासवान": "Paswan", "गुप्ता": "Gupta", "पटेल": "Patel",
    "राम": "Ram", "बाई": "Bai", "बेगम": "Begum", "खातून": "Khatoon",
    "रामपुर": "Rampur", "बेलवा": "Belwa", "पिपरा": "Pipra", "किशनपुर": "Kishanpur",
    "माधोपुर": "Madhopur", "पुरानी": "Purani", "नई": "Nayi", "बस्ती": "Basti",
    "हरिजन": "Harijan", "दक्षिण": "Dakshin", "टोला": "Tola",
    "पति": "pati", "पत्नी": "patni", "मरीज": "marij", "मरीज़": "marij",
    "महिला": "mahila", "नाम": "naam", "गलत": "galat", "सही": "sahi",
    "है": "hai", "हैं": "hain", "का": "ka", "की": "ki", "के": "ke",
    "से": "se", "में": "mein", "घर": "ghar", "अरे": "are", "सर": "sir",
    "जी": "ji", "दीदी": "didi", "नहीं": "nahi", "हाँ": "haan", "हां": "haan",
}

_DEV_VOWELS: dict[str, str] = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}
_DEV_CONSONANTS: dict[str, str] = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
}
_DEV_MATRAS: dict[str, str] = {
    "ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
}
_NAME_STOPWORDS = {
    "didi", "ji", "sir", "are", "arre", "naam", "name", "hai", "hain", "is",
    "patient", "marij", "patni", "pati", "husband", "wife", "spouse", "guardian",
    "mahila", "aurat", "ka", "ki", "ke", "se", "mein", "main", "ghar", "hoon",
    "galat", "sahi", "wrong", "correct", "nahi", "nahin", "no", "yes", "haan",
    "han", "unke", "unka", "unki", "uska", "uski", "aur", "ya", "theek",
    "badlo", "change", "dobara", "batao", "batayein", "w/o", "wo",
}


def _transliterate_devanagari_word(word: str) -> str:
    if word in _DEVANAGARI_WORD_MAP:
        return _DEVANAGARI_WORD_MAP[word]
    if not any("\u0900" <= ch <= "\u097f" for ch in word):
        return word
    out: list[str] = []
    n = len(word)
    i = 0
    while i < n:
        ch = word[i]
        if ch in _DEV_VOWELS:
            out.append(_DEV_VOWELS[ch])
            i += 1
        elif ch in _DEV_CONSONANTS:
            base = _DEV_CONSONANTS[ch]
            # Handle nukta combined characters (e.g., ड + ़)
            if i + 1 < n and word[i + 1] == "़":
                combined = ch + "़"
                base = _DEV_CONSONANTS.get(combined, base)
                i += 1
            if i + 1 < n and word[i + 1] == "्":
                out.append(base)
                i += 2
            elif i + 1 < n and word[i + 1] in _DEV_MATRAS:
                out.append(base + _DEV_MATRAS[word[i + 1]])
                i += 2
            else:
                # Word-final schwa deletion in Hindi (e.g. अभिनव -> Abhinav, राज -> Raj)
                is_word_end = (i == n - 1) or (i == n - 2 and word[i + 1] in ("ं", "ँ"))
                out.append(base if is_word_end and i > 0 else base + "a")
                i += 1
        elif ch in ("ं", "ँ"):
            out.append("n")
            i += 1
        elif ch == "ः":
            out.append("h")
            i += 1
        elif ch in _DEV_MATRAS:
            out.append(_DEV_MATRAS[ch])
            i += 1
        else:
            i += 1
    return "".join(out)


def _transliterate_indic_block(text: str, scheme: str) -> str:
    try:
        from indic_transliteration import sanscript

        src = sanscript.TAMIL if scheme == "tamil" else sanscript.TELUGU
        roman = sanscript.transliterate(text, src, sanscript.ITRANS)
        ascii_text = unicodedata.normalize("NFKD", roman).encode("ascii", "ignore").decode("ascii")
        return ascii_text.lower()
    except Exception:
        return text


def transliterate_to_english(text: str) -> str:
    """Convert Devanagari, Tamil, or Telugu script in `text` to Roman English equivalents."""
    if not text:
        return ""
    out = re.sub(r"[\u0900-\u097f]+", lambda m: _transliterate_devanagari_word(m.group(0)), text)
    out = re.sub(r"[\u0b80-\u0bff]+", lambda m: _transliterate_indic_block(m.group(0), "tamil"), out)
    out = re.sub(r"[\u0c00-\u0c7f]+", lambda m: _transliterate_indic_block(m.group(0), "telugu"), out)
    return out


def _clean_person_name(raw_name: str | None) -> str | None:
    """Transliterate to English, strip non-name stop words ('galat', 'pati', 'hai', etc.), and Title-Case."""
    if not raw_name:
        return None
    roman = transliterate_to_english(raw_name)
    tokens = [
        tok
        for tok in re.findall(r"[A-Za-z]+", roman)
        if tok.lower() not in _NAME_STOPWORDS and len(tok) >= 2
    ]
    if not tokens:
        return None
    cleaned = " ".join(tokens[:4]).title()
    return cleaned if 2 <= len(cleaned) <= 45 else None


_AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
_PHONE_RE = re.compile(r"(?<!\d)(?:(?:\+91|0)[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")

_PATIENT_FIELDS = frozenset({"full_name", "spouse_or_guardian_name", "age_years", "village", "tola"})
_VISIT_FIELDS = frozenset({"gestational_weeks", "bp_systolic", "bp_diastolic", "hb_g_dl", "symptoms"})
_VALID_SYMPTOM_CODES = frozenset(code.value for code in SymptomCode)
_VALID_PRESENCE = frozenset(presence.value for presence in Presence)
_VALID_CONFIRMATION = frozenset(intent.value for intent in ConfirmationIntent)

_EVON_JSON_SCHEMA_PROMPT = (
    "\nReturn ONLY a single valid JSON object (no markdown, no explanation) with exact top-level keys:\n"
    '{"patient": {"full_name": string|null, "spouse_or_guardian_name": string|null, '
    '"age_years": int|null, "village": string|null, "tola": string|null}, '
    '"visit": {"gestational_weeks": int|null, "bp_systolic": int|null, "bp_diastolic": int|null, '
    '"hb_g_dl": float|null, "symptoms": [{"code": SymptomCode, "presence": "PRESENT"|"ABSENT"|"UNKNOWN", '
    '"phrase": string|null}]}, "confirmation_intent": "YES"|"NO"|"UNCLEAR"}.\n'
    f"Allowed SymptomCode values: {', '.join(code.value for code in SymptomCode)}."
)


def redact_pii(text: str) -> str:
    """Mask 12-digit Aadhaar and 10-digit Indian mobile numbers before external LLM calls."""
    if not text:
        return ""
    redacted = _AADHAAR_RE.sub("[REDACTED_AADHAAR]", text)
    return _PHONE_RE.sub("[REDACTED_PHONE]", redacted)


class _ModelOutput(ProjectModel):
    patient: PatientReference = Field(default_factory=PatientReference)
    visit: ExtractedVisit = Field(default_factory=ExtractedVisit)
    confirmation_intent: ConfirmationIntent = ConfirmationIntent.UNCLEAR


def _parse_evon_output(raw_text: str) -> _ModelOutput:
    """Strip reasoning tags/fences, prune unknown keys, and validate into `_ModelOutput`."""
    cleaned_text = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.S | re.I).strip()
    cleaned_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned_text, flags=re.I).strip()
    start = cleaned_text.find("{")
    end = cleaned_text.rfind("}")
    if start != -1 and end > start:
        cleaned_text = cleaned_text[start:end + 1]
    payload = json.loads(cleaned_text)
    if not isinstance(payload, dict):
        raise ValueError("Evon output must be a JSON object")

    raw_patient = payload.get("patient")
    patient_dict: dict[str, Any] = {}
    if isinstance(raw_patient, dict):
        for key in _PATIENT_FIELDS:
            if key in raw_patient:
                val = raw_patient[key]
                patient_dict[key] = None if (key == "age_years" and val == "") else val

    raw_visit = payload.get("visit")
    visit_dict: dict[str, Any] = {}
    if isinstance(raw_visit, dict):
        for key in _VISIT_FIELDS - {"symptoms"}:
            if key in raw_visit:
                val = raw_visit[key]
                visit_dict[key] = None if val == "" else val
        raw_symptoms = raw_visit.get("symptoms")
        if isinstance(raw_symptoms, list):
            clean_symptoms: list[dict[str, Any]] = []
            for item in raw_symptoms:
                if not isinstance(item, dict):
                    continue
                raw_code = item.get("code")
                code_val = raw_code.strip().upper() if isinstance(raw_code, str) else ""
                if code_val not in _VALID_SYMPTOM_CODES:
                    continue
                raw_presence = item.get("presence")
                if raw_presence is None or (isinstance(raw_presence, str) and not raw_presence.strip()):
                    presence_val = Presence.PRESENT.value
                elif isinstance(raw_presence, str):
                    presence_val = raw_presence.strip().upper()
                else:
                    continue
                if presence_val not in _VALID_PRESENCE:
                    continue
                entry: dict[str, Any] = {"code": code_val, "presence": presence_val}
                if "phrase" in item and (item["phrase"] is None or isinstance(item["phrase"], str)):
                    entry["phrase"] = item["phrase"]
                clean_symptoms.append(entry)
            visit_dict["symptoms"] = clean_symptoms

    raw_ci = payload.get("confirmation_intent")
    ci_val = raw_ci.strip().upper() if isinstance(raw_ci, str) else ""
    confirmation_val = ci_val if ci_val in _VALID_CONFIRMATION else ConfirmationIntent.UNCLEAR.value

    return _ModelOutput.model_validate(
        {
            "patient": patient_dict,
            "visit": visit_dict,
            "confirmation_intent": confirmation_val,
        }
    )


class _EvonStructuredModel:
    """OpenAI-compatible async adapter for Gnani Evon v3.3 with thinking disabled."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "EMPTY",
        model: str = "gnani/gnani-evon-v3.3-30B-A3B",
        timeout_s: float = 8.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.model = model
        self.timeout_s = timeout_s

    async def ainvoke(self, messages: list[tuple[str, str]]) -> _ModelOutput:
        oai_messages: list[dict[str, str]] = []
        for role, content in messages:
            oai_role = "user" if role == "human" else role
            if oai_role == "system":
                content = f"{content}\n{_EVON_JSON_SCHEMA_PROMPT}"
            oai_messages.append({"role": oai_role, "content": content})

        request_payload = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": 0.0,
            "max_tokens": 384,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=request_payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        content = str(data["choices"][0]["message"]["content"])
        return _parse_evon_output(content)


def _merged_lexicon_list(lang: str, key: str) -> list[str]:
    items = list(lexicon(lang).get(key, []))
    if lang != DEFAULT_LANGUAGE:
        items.extend(lexicon(DEFAULT_LANGUAGE).get(key, []))
    return list(dict.fromkeys(items))


def _merged_symptom_terms(lang: str) -> dict[SymptomCode, list[str]]:
    primary = lexicon(lang).get("symptoms", {})
    fallback = lexicon(DEFAULT_LANGUAGE).get("symptoms", {}) if lang != DEFAULT_LANGUAGE else {}
    out: dict[SymptomCode, list[str]] = {}
    for code in SymptomCode:
        terms = list(primary.get(code.value, []))
        terms.extend(fallback.get(code.value, []))
        out[code] = list(dict.fromkeys(terms))
    return out


def _normalize_numbers(text: str, language: str = DEFAULT_LANGUAGE) -> str:
    result = text.lower()
    merged: dict[str, str] = {}
    if language != DEFAULT_LANGUAGE:
        merged.update(lexicon(DEFAULT_LANGUAGE).get("numbers", {}))
    merged.update(lexicon(language).get("numbers", {}))
    for phrase, digits in sorted(merged.items(), key=lambda pair: -len(pair[0])):
        result = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", digits, result)
    return re.sub(r"(\d+)\s*(?:point|dot|dashamlav|pulli)\s*(\d+)", r"\1.\2", result)


def _fallback_extract(
    transcript: str,
    expected_action: str | None,
    *,
    language: str = DEFAULT_LANGUAGE,
) -> TurnExtraction:
    """Conservative local extraction for outages and development without API keys."""
    lang = normalize_language_code(language)
    normalized = _normalize_numbers(transcript, lang)
    visit_data: dict[str, Any] = {}
    patient = PatientReference()

    bp = re.search(r"(?:bp|blood pressure|बीपी|பிபி|బీపీ)?\s*(\d{2,3})\s*(?:/|by|bai|over|बाई|बाय|और|பாய்|బై)\s*(\d{2,3})", normalized)
    if bp:
        visit_data.update(bp_systolic=int(bp.group(1)), bp_diastolic=int(bp.group(2)))
    weeks = re.search(r"(\d{1,2})\s*(?:weeks?|hafte|hafta|saptah|vaaram|varam|varalu|हफ्ते|हफ़्ते|सप्ताह|வாரம்|వారాలు|వారం)", normalized)
    if weeks and int(weeks.group(1)) > 0:
        visit_data["gestational_weeks"] = int(weeks.group(1))
    hb = re.search(r"(?:\bhb\b|hemoglobin|haemoglobin|हीमोग्लोबिन|एचबी|ஹீமோகுளோபின்|హీమోగ్లోబిన్)\s*(?:hai|is|है|உள்ளது|ఉంది)?\s*(\d{1,2}(?:\.\d+)?)", normalized)
    if hb:
        visit_data["hb_g_dl"] = float(hb.group(1))

    neg_words = _merged_lexicon_list(lang, "negation_words") or ["nahi", "nahin", "no", "not"]
    neg_alt = "|".join(re.escape(w) for w in neg_words)
    symptoms: list[SymptomReport] = []
    for code, terms in _merged_symptom_terms(lang).items():
        found = next((term for term in terms if term in normalized), None)
        if found:
            start = normalized.find(found)
            prefix = normalized[max(0, start - 18):start]
            suffix = normalized[start + len(found):start + len(found) + 20]
            negated = bool(
                re.search(rf"(?:{neg_alt})\s*$", prefix)
                or re.match(rf"\s*(?:{neg_alt})(?:\s|$)", suffix)
            )
            symptoms.append(SymptomReport(code=code, presence=Presence.ABSENT if negated else Presence.PRESENT, phrase=found))

    yes_terms = _merged_lexicon_list(lang, "yes")
    no_terms = _merged_lexicon_list(lang, "no")
    yes_re = re.compile("|".join(rf"\b{re.escape(w)}\b" if w.isascii() else re.escape(w) for w in yes_terms), re.I)
    no_re = re.compile("|".join(rf"\b{re.escape(w)}\b" if w.isascii() else re.escape(w) for w in no_terms), re.I)

    denial_terms = _merged_lexicon_list(lang, "general_denial")
    denial_re = re.compile("|".join(re.escape(w) for w in denial_terms), re.I) if denial_terms else re.compile(r"(?!)")
    # A clear general denial records known symptoms as absent, never as present.
    general_denial = bool(
        denial_re.search(normalized)
        or (expected_action == "ASK_SYMPTOMS" and no_re.search(transcript) and not yes_re.search(transcript))
    )
    if not symptoms and general_denial:
        symptoms = [SymptomReport(code=code, presence=Presence.ABSENT) for code in SymptomCode]
    if symptoms:
        visit_data["symptoms"] = symptoms

    roman_transcript = transliterate_to_english(transcript)
    transcript_for_patient = roman_transcript

    kw_spouse = list(dict.fromkeys([*lexicon(lang).get("keywords", {}).get("spouse", []), *lexicon(DEFAULT_LANGUAGE).get("keywords", {}).get("spouse", [])]))
    kw_patient = list(dict.fromkeys([*lexicon(lang).get("keywords", {}).get("patient", []), *lexicon(DEFAULT_LANGUAGE).get("keywords", {}).get("patient", [])]))
    spouse_alt = "|".join(re.escape(w) for w in kw_spouse if w.isascii()) or "pati|husband|spouse"
    patient_alt = "|".join(re.escape(w) for w in kw_patient if w.isascii()) or "marij|patient|mahila|aurat"

    # Extract explicit spouse/husband name clauses (skipping clauses like 'pati ka naam galat hai')
    _name_stop_lookahead = (
        r"(?=\s+(?:\d|\b(?:hai|hain|aur|age|umra|weeks?|hafte|hafta|saptah|vaaram|varam|bp|hb|hemoglobin|gaon|oor|ooru|pati|patni|marij|patient)\b)|[.,;!?]|$)"
    )
    for spouse_match in re.finditer(
        rf"\b(?:patni|{spouse_alt}|w/o|wife\s+of|guardian)\b(?:\s+ka\s+naam|\s+name|\s+peru|\s+peyar)?\s*(?:is|hai|:)?\s*([A-Za-z]+(?:\s+[A-Za-z]+){{0,4}}?)"
        + _name_stop_lookahead,
        roman_transcript,
        re.I,
    ):
        cleaned_spouse = _clean_person_name(spouse_match.group(1))
        if cleaned_spouse:
            patient.spouse_or_guardian_name = cleaned_spouse
            transcript_for_patient = transcript_for_patient.replace(spouse_match.group(0), " ")

    # Extract explicit patient/wife name clauses (e.g. 'marij ka naam Sunita Devi hai')
    for pat_match in re.finditer(
        rf"\b(?:{patient_alt})\b(?:\s+ka\s+naam|\s+name|\s+peru|\s+peyar)?\s*(?:is|hai|:)?\s*([A-Za-z]+(?:\s+[A-Za-z]+){{0,4}}?)"
        + _name_stop_lookahead,
        transcript_for_patient,
        re.I,
    ):
        cleaned_pat = _clean_person_name(pat_match.group(1))
        if cleaned_pat:
            patient.full_name = cleaned_pat

    mentions_patient_keyword = bool(re.search(rf"\b(?:{patient_alt})\b", roman_transcript, re.I))
    is_spouse_only_turn = expected_action == "ASK_SPOUSE_NAME" and not mentions_patient_keyword

    if not is_spouse_only_turn and patient.full_name is None:
        name = re.search(
            r"\b([A-Za-z]+\s+(?:devi|kumari|bai|begum|yadav|verma|sharma|singh|khatoon|murugan|reddy|rao|sundaram))\b",
            transcript_for_patient,
            re.I,
        )
        if name:
            cleaned_pat = _clean_person_name(name.group(1))
            if cleaned_pat and cleaned_pat != patient.spouse_or_guardian_name:
                patient.full_name = cleaned_pat

    for token, canonical in (
        ("rampur", "Rampur"), ("रामपुर", "Rampur"),
        ("belwa", "Belwa"), ("बेलवा", "Belwa"),
        ("madurai", "Madurai"), ("மதுரை", "Madurai"),
        ("melur", "Melur"), ("மேலூர்", "Melur"),
        ("guntur", "Guntur"), ("గుంటూరు", "Guntur"),
        ("tenali", "Tenali"), ("తెనాలి", "Tenali"),
    ):
        if token in normalized or token in roman_transcript.lower():
            patient.village = canonical
            break
    for token, canonical in (
        ("purani basti", "Purani Basti"), ("पुरानी बस्ती", "Purani Basti"),
        ("nayi basti", "Nayi Basti"), ("नई बस्ती", "Nayi Basti"),
        ("harijan tola", "Harijan Tola"), ("dakshin tola", "Dakshin Tola"),
    ):
        if token in normalized or token in roman_transcript.lower():
            patient.tola = canonical
            break
    age = re.search(r"\b(?:age|umra|vayasu)\s*(\d{2})\b|\b(\d{2})\s*(?:saal|years|साल)\b", normalized)
    if age:
        patient.age_years = int(age.group(1) or age.group(2))

    if expected_action == "ASK_PATIENT_NAME" and patient.full_name is None and not visit_data:
        cleaned_pat = _clean_person_name(transcript_for_patient)
        if cleaned_pat:
            patient.full_name = cleaned_pat
    if expected_action == "ASK_SPOUSE_NAME" and patient.spouse_or_guardian_name is None and not visit_data:
        cleaned_spouse = _clean_person_name(roman_transcript)
        if cleaned_spouse:
            patient.spouse_or_guardian_name = cleaned_spouse
    if expected_action == "ASK_WEEKS" and "gestational_weeks" not in visit_data:
        bare = re.search(r"\b(\d{1,2})\b", normalized)
        if bare and int(bare.group(1)) > 0:
            visit_data["gestational_weeks"] = int(bare.group(1))
    if expected_action == "ASK_HB" and "hb_g_dl" not in visit_data:
        bare = re.search(r"\b(\d{1,2}(?:\.\d+)?)\b", normalized)
        if bare:
            visit_data["hb_g_dl"] = float(bare.group(1))

    has_values = bool(visit_data) or any(patient.model_dump().values())
    confirmation = ConfirmationIntent.UNCLEAR
    if expected_action in {
        "CONFIRM_VALUES", "ASK_ANOTHER_PATIENT", "CONFIRM_PATIENT_MATCH",
        "ASK_READ_PRIOR_RECORD",
    } and not has_values:
        if yes_re.search(transcript):
            confirmation = ConfirmationIntent.YES
        elif no_re.search(transcript):
            confirmation = ConfirmationIntent.NO
    return TurnExtraction(patient=patient, visit=ExtractedVisit.model_validate(visit_data), confirmation_intent=confirmation)


def _build_language_prompt_addon(language: str) -> str:
    lang = normalize_language_code(language)
    if lang == DEFAULT_LANGUAGE:
        return ""
    meta = locale_meta(lang)
    lang_name = meta.get("name_en", lang)
    sym_map = lexicon(lang).get("symptoms", {})
    hints = "; ".join(f"{code}: {', '.join(terms[:4])}" for code, terms in sym_map.items() if terms)
    return (
        f"\nActive caller language: {lang_name}. The utterance may be in {lang_name} native script, "
        f"Romanized {lang_name} (e.g. Tanglish/Telglish), or code-mixed with English. "
        f"Transliterate all patient/spouse/village names into English Roman Title Case. "
        f"Local symptom terms -> schema codes: {hints}."
    )


async def extract_turn(
    transcript: str,
    expected_action: str | None = None,
    *,
    language: str = DEFAULT_LANGUAGE,
    llm: Any | None = None,
) -> TurnExtraction:
    """Extract one utterance asynchronously; try Evon v3.3 -> Groq -> local regex fallback."""
    lang = normalize_language_code(language)
    fallback = _fallback_extract(transcript, expected_action, language=lang)
    models: list[Any] = []
    if llm is None:
        evon_base_url = os.getenv("EVON_BASE_URL", "").strip()
        if evon_base_url:
            try:
                timeout_s = float(os.getenv("EVON_TIMEOUT_S", "8.0").strip() or "8.0")
            except ValueError:
                timeout_s = 8.0
            models.append(
                _EvonStructuredModel(
                    base_url=evon_base_url,
                    api_key=os.getenv("EVON_API_KEY", "EMPTY").strip() or "EMPTY",
                    model=os.getenv("EVON_MODEL", "gnani/gnani-evon-v3.3-30B-A3B").strip()
                    or "gnani/gnani-evon-v3.3-30B-A3B",
                    timeout_s=timeout_s,
                )
            )
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if api_key:
            try:
                from langchain_groq import ChatGroq
            except Exception as exc:
                logger.warning("Could not load Groq integration; skipping Groq provider: %s", exc)
            else:
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
        if not models:
            return fallback
    else:
        models = [llm]

    context = f"Previous prompt/action: {expected_action or 'none'}. Interpret a short answer using this context."
    sys_prompt = SYSTEM_PROMPT + _build_language_prompt_addon(lang) + "\n" + context
    messages = [("system", sys_prompt), ("human", redact_pii(transcript))]
    for model in models:
        try:
            result = await model.ainvoke(messages)
            parsed = result if isinstance(result, _ModelOutput) else _ModelOutput.model_validate(result)
            roman_utt = transliterate_to_english(transcript)
            mentions_spouse = bool(re.search(r"\b(pati|husband|spouse)\b", roman_utt, re.I))
            mentions_patient = bool(re.search(r"\b(marij|patient|mahila|aurat)\b", roman_utt, re.I))
            if expected_action == "ASK_SPOUSE_NAME" and not mentions_patient:
                if parsed.patient.full_name and not parsed.patient.spouse_or_guardian_name:
                    parsed.patient.spouse_or_guardian_name = parsed.patient.full_name
                parsed.patient.full_name = None
            elif mentions_spouse and not mentions_patient and not fallback.patient.full_name:
                if parsed.patient.full_name and not parsed.patient.spouse_or_guardian_name:
                    parsed.patient.spouse_or_guardian_name = parsed.patient.full_name
                parsed.patient.full_name = None
            if parsed.visit.gestational_weeks == 0:
                parsed.visit.gestational_weeks = None
            merged_patient = _merge_patient(parsed.patient, fallback.patient)
            return TurnExtraction(
                patient=merged_patient,
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
    if values.get("full_name"):
        values["full_name"] = _clean_person_name(values["full_name"])
    if values.get("spouse_or_guardian_name"):
        values["spouse_or_guardian_name"] = _clean_person_name(values["spouse_or_guardian_name"])
    if values.get("village"):
        values["village"] = transliterate_to_english(values["village"]).strip().title()
    if values.get("tola"):
        values["tola"] = transliterate_to_english(values["tola"]).strip().title()
    return PatientReference.model_validate(values)


def _merge_visit(primary: ExtractedVisit, fallback: ExtractedVisit) -> ExtractedVisit:
    values = primary.model_dump()
    fallback_values = fallback.model_dump()
    for key, value in fallback_values.items():
        if values.get(key) in (None, []) and value not in (None, []):
            values[key] = value
    return ExtractedVisit.model_validate(values)
