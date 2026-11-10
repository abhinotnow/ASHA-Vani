- [ ] maybe reach out to an actual doctor/some source online to verify if our schema is good? not super necessary for a demo but good to keep in mind.
- [x] check if the call is automatically ending cleanly after confirmation (`COMMITTED_VERIFIED` / `END_CALL` -> hangup / close `/ws/call` and `/ws/exotel` after final TTS completes).
- [x] check if the physical phone line (Exotel PSTN bridge) is actually working end-to-end (`/ws/exotel` protocol, Basic Auth, 8 kHz PCM media chunking, mark acks, and continuous WAV recording verified; ready for live phone dial test).
- [x] full all-in-one call recording & in-app audio player (`.mp4` / `.webm` / `.wav`): mix live mic + Timbre TTS + DTMF into one continuous recording (`MediaStreamDestination` + `MediaRecorder` and `/api/recordings/{session_id}`) with a replay bar in the inspection sheet.

## Edge Cases Fixed
- [x] not taking case in which weeks is given as `0` properly for pregnancy or when the patient is not pregnant (`0`, `0 weeks`, `0 hafte`, `shunya`, `nahi` at `ASK_WEEKS` now cleanly sets `is_pregnant=False`, clears `gestational_weeks=None` before validation, and transitions to `ASK_BP` -> `ASK_HB` -> `ASK_SYMPTOMS`).
- [x] "not a diagnosis" phrasing / disclaimer cleanup in spoken prompts (`reply_text` no longer repeats `"yeh diagnosis nahi hai"` on mid-call questions, readbacks, or confirmations).
- [x] rule stuff again (`triage.py` now fires `R2` `URGENT` on standalone severe symptoms like `EPIGASTRIC_PAIN` / `SEVERE_HEADACHE` / `BLURRED_VISION` even when BP is normal, and `R4` `URGENT` on bleeding when `weeks is None`; raw rule codes removed from spoken TTS).
- [x] repeating the same stuff (eliminated mid-call history confirmation loops and allowed new patient facts at `ASK_ANOTHER_PATIENT` to start the next visit directly).
- [x] database stuff (Supabase `visits` + `call_events` hydration in `hydrateSavedRegister()` now restores both wife and husband names and full visit rows on refresh).
- [x] interrupting speaker (`HalfDuplexGate` now holds server mic mute for the full real-time playback duration of streamed TTS audio, drains stale `audio_queue` frames while muted, and never resets `playbackCursor` backwards in the browser).
- [x] not saving the right name of the wife and husband (`extraction.py` and `graph.py` `merge_turn` now route names during `ASK_SPOUSE_NAME` strictly to `patient.spouse_or_guardian_name` and strip the spouse clause before matching `patient.full_name`).
- [x] not talking about / asking for features like stomach pain (`EPIGASTRIC_PAIN` / `pet dard` / `pet mein dard` / `stomach pain`) and other danger signs explicitly in the spoken prompts and readback.
- [x] too many confirmations (`sm confirmations` — `_lookup_history` now links existing patients silently in the background without interrupting the call with `CONFIRM_PATIENT_MATCH` and `ASK_READ_PRIOR_RECORD`).


