# ASHA-Vani: 1-Page Solution Pitch

> **A multilingual phone call powered by Gnani Prisma STT, Gnani Evon v3.3 (`30B-A3B`), and Gnani Timbre v2.5 TTS that turns any rural village home visit into a confirmed record for the PHC.**

---

## 1. The Problem: Paper Registers Delay PHC Awareness
Nearly **one million ASHA workers** are the first point of contact for rural families across India. Visit notes still start on paper, and mobile apps assume smartphone ownership, typing fluency, and reliable data. When a worker hears a maternal warning sign or abnormal vital in the field, that finding can wait days before reaching a Primary Health Centre (**PHC**) Medical Officer.

---

## 2. The Solution: Speak Into Any Feature Phone
Built for **Pan-India rural healthcare**, **ASHA-Vani** lets any worker dial in from a basic 2G feature phone and speak naturally in **Hindi, Tamil, or Telugu** (including code-mixed **Hinglish, Tanglish, and Telglish**), backed by drop-in JSON locales ready to scale across **all 10 Gnani Indian languages**:

1. **Listen (`Gnani Prisma v3.0 STT`):** Streams 8 kHz telephone audio through real-time Energy VAD gating and `indic-auto` language detection so the worker is never cut off mid-sentence.
2. **Understand (`Gnani Evon v3.3 · 30B-A3B`):** Redacts phone and Aadhaar numbers pre-LLM, extracts patient details, vitals (e.g., *"dedh sau by pachhanave"* -> `150/95`), and symptoms with `enable_thinking=False`, and transliterates names into clean **English Title Case** for PHC doctors (with automatic Groq Llama 3.3 and regex failover).
3. **Check & Read Back (`Gnani Timbre v2.5 TTS`):** Validates implausible readings before speaking back captured values digit-by-digit in the worker's native voice (`Poorvi` Hindi, `Asmita` Tamil, `Suhana` Telugu) under strict half-duplex anti-echo gating.
4. **Confirm & Share (`PHC Register Console`):** The ASHA confirms or corrects details aloud. Visits stay marked `UNVERIFIED` during the call and flip to `VERIFIED` only after explicit confirmation, alongside the full-call WAV audio archive.

---

## 3. Multilingual by Design & Built for Safety
- **Zero-Menu Language Routing:** Unknown handsets hear a trilingual greeting (*"Namaste! Vanakkam! Namaskaram!"*), auto-resolve **Hindi (`hi-IN`), Tamil (`ta-IN`), or Telugu (`te-IN`)**, and can switch languages anytime via **Keypad `9`**.
- **Human-in-the-Loop Decision Support:** Unclear values trigger a polite repeat instead of guessing, and PHC clinicians always own follow-up and referral decisions.

---

## 4. Future Prospects & Rural Scalability
1. **Household Medicine Kit Tracking:** Track IFA, ORS, Zinc, Calcium, and TT supplies dispensed per household caretaker to prevent village stockouts.
2. **Automated Monthly Due-Lists & HBNC:** Auto-compile ANC checkup and immunization due-lists alongside 6-visit Home-Based Newborn Care follow-ups.
3. **Cluster PHC Deployment & SMS Receipts:** One central PHC dashboard serves 10–25 villages while verified calls trigger local-script SMS receipts for ASHA performance incentives.

---

**Built with Gnani.ai** (`Prisma v3.0 STT` · `Evon v3.3 30B-A3B` · `Timbre v2.5 TTS`) for the **Great Indian AI Internship Challenge**.

`#GreatIndianAIInternshipChallenge` `#GreatIndianInternship` `#GnaniAI` `#BuildForIndia` `#AIForStudents` `#VoiceAI` `#SovereignAI`
