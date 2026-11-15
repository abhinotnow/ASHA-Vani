# Multilingual Prompt Review Sheet (HI | TA | TE)

Side-by-side spoken prompts for native-speaker review before live demo. Clinical terms (BP, Hemoglobin, weeks, screening, PHC) are kept in English intentionally.

## Prompts

| Key | Hindi / Hinglish (hi) | Tamil (	a) | Telugu (	e) |
|---|---|---|---|
| greeting | Namaste Didi! ASHA-Vani mein aapka swagat hai. Aap kis gaon mein hain aur kis marij ke ghar se bol rahi hain? | வணக்கம் அக்கா! ASHA-Vani-க்கு வரவேற்கிறோம். நீங்கள் எந்த ஊரில் இருக்கிறீர்கள், எந்த patient வீட்டிலிருந்து பேசுகிறீர்கள்? | నమస్కారం అక్క! ASHA-Vani కు స్వాగతం. మీరు ఏ గ్రామంలో ఉన్నారు, ఏ patient ఇంటి నుండి మాట్లాడుతున్నారు? |
| greeting_known_worker | Namaste {short_name} Didi ({asha_id})! ASHA-Vani mein aapka swagat hai. Marij (patient) ka poora naam aur gaon batayein. | வணக்கம் {short_name} அக்கா ({asha_id})! ASHA-Vani-க்கு வரவேற்கிறோம். Patient-ன் முழு பெயரும் ஊரும் சொல்லுங்கள். | నమస్కారం {short_name} అక్క ({asha_id})! ASHA-Vani కు స్వాగతం. Patient పూర్తి పేరు మరియు గ్రామం చెప్పండి. |
| greet_worker_ask_patient | Ji {short_name} Didi ({asha_id}), marij (patient) ka poora naam aur gaon batayein. | சரி {short_name} அக்கா ({asha_id}), patient-ன் முழு பெயரும் ஊரும் சொல்லுங்கள். | సరే {short_name} అక్క ({asha_id}), patient పూర్తి పేరు మరియు గ్రామం చెప్పండి. |
| ask_asha_id | Kripya apna 3-digit ASHA worker ID batayein. | உங்கள் 3-digit ASHA worker ID சொல்லுங்கள். | మీ 3-digit ASHA worker ID చెప్పండి. |
| ASK_PATIENT_NAME | Didi, marij (patient) ka poora naam aur gaon batayein. | அக்கா, patient-ன் முழு பெயரும் ஊரும் சொல்லுங்கள். | అక్క, patient పూర్తి పేరు మరియు గ్రామం చెప్పండి. |
| ASK_SPOUSE_NAME | Ji Didi, pati (husband) ka naam kya hai? | சரி அக்கா, கணவர் (husband) பெயர் என்ன? | సరే అక్క, భర్త (husband) పేరు ఏమిటి? |
| ASK_WEEKS | Theek hai. Unke pregnancy ke kitne hafte (weeks) hue hain? Agar garbhvati nahi hain toh zero batayein. | சரி. Pregnancy எத்தனை weeks ஆகிறது? கர்ப்பம் இல்லை என்றால் zero என்று சொல்லுங்கள். | సరే. Pregnancy ఎన్ని weeks అయ్యింది? గర్భవతి కాకపోతే zero అని చెప్పండి. |
| ASK_BP | Ji, unka BP (blood pressure) kitna aaya hai? Systolic aur diastolic dono number batayein. | சரி, அவர்களுக்கு BP (blood pressure) எவ்வளவு உள்ளது? Systolic மற்றும் diastolic இரண்டு எண்ணையும் சொல்லுங்கள். | సరే, వారి BP (blood pressure) ఎంత ఉంది? Systolic మరియు diastolic రెండు నంబర్లు చెప్పండి. |
| ASK_BP_NON_PREGNANT | Ji Didi, Non-Pregnant checkup note ho gaya. Unka BP (blood pressure) kitna aaya hai? Systolic aur diastolic dono number batayein. | சரி அக்கா, Non-Pregnant checkup குறிக்கப்பட்டது. அவர்களுக்கு BP (blood pressure) எவ்வளவு உள்ளது? Systolic மற்றும் diastolic இரண்டு எண்ணையும் சொல்லுங்கள். | సరే అక్క, Non-Pregnant checkup నమోదు అయ్యింది. వారి BP (blood pressure) ఎంత ఉంది? Systolic మరియు diastolic రెండు నంబర్లు చెప్పండి. |
| ASK_HB | Unka Hemoglobin (Hb) kitna aaya hai, aur koi takleef ya lakshan hai kya? | அவர்களின் Hemoglobin (Hb) எவ்வளவு உள்ளது, வேறு ஏதாவது அறிகுறி அல்லது சிரமம் உள்ளதா? | వారి Hemoglobin (Hb) ఎంత ఉంది, ఏదైనా ఇబ్బంది లేదా లక్షణాలు ఉన్నాయా? |
| ASK_SYMPTOMS | Kya unhe pet mein dard, tez sar dard, dhundhla dikhna, bleeding, daura, soojan, paani aana, ya bachche ki halchal kam lagna hai? | அவர்களுக்கு வயிற்று வலி, கடுமையான தலைவலி, மங்கலான பார்வை, bleeding, வலிப்பு, வீக்கம், நீர் கசிவு, அல்லது குழந்தையின் அசைவு குறைவு உள்ளதா? | వారికి కడుపు నొప్పి, తీవ్రమైన తలనొప్పి, మసక చూపు, bleeding, ఫిట్స్, వాపు, నీరు కారడం, లేదా బిడ్డ కదలికలు తగ్గడం ఉన్నాయా? |
| confirm_routine | Dhanyavaad Didi. {label} ka record confirm ho gaya hai. Kya aap kisi aur mahila ka record dena chahti hain? | நன்றி அக்கா. {label} பதிவு confirm செய்யப்பட்டது. வேறு ஒரு பெண்ணின் தகவலை கொடுக்க வேண்டுமா? | ధన్యవాదాలు అక్క. {label} రికార్డు confirm అయ్యింది. ఇంకో మహిళ వివరాలు ఇవ్వాలా? |
| confirm_escalate | Dhanyavaad Didi. {label} ka record confirm ho gaya hai. Screening {level} hai, kripya PHC team se turant sampark karein. Kya aap kisi aur mahila ka record dena chahti hain? | நன்றி அக்கா. {label} பதிவு confirm செய்யப்பட்டது. Screening {level} ஆக உள்ளது, உடனே PHC குழுவைத் தொடர்பு கொள்ளவும். வேறு ஒரு பெண்ணின் தகவலை கொடுக்க வேண்டுமா? | ధన్యవాదాలు అక్క. {label} రికార్డు confirm అయ్యింది. Screening {level} గా ఉంది, వెంటనే PHC టీమ్ ని సంప్రదించండి. ఇంకో మహిళ వివరాలు ఇవ్వాలా? |
| next_patient | Theek hai Didi. Ab agle marij ka poora naam aur gaon batayein. | சரி அக்கா. அடுத்த patient-ன் முழு பெயரும் ஊரும் சொல்லுங்கள். | సరే అక్క. తర్వాతి patient పూర్తి పేరు మరియు గ్రామం చెప్పండి. |
| finish_call | Dhanyavaad Didi. ASHA-Vani call ab band ho rahi hai. Namaste. | நன்றி அக்கா. ASHA-Vani அழைப்பு இப்போது முடிகிறது. வணக்கம். | ధన్యవాదాలు అక్క. ASHA-Vani కాల్ ఇప్పుడు ముగుస్తోంది. నమస్కారం. |
| continue_prompt | Didi, kya aap kisi aur garbhavati mahila ka record dena chahti hain? Haan ya nahi batayein. | அக்கா, வேறு ஒரு கர்ப்பிணிப் பெண்ணின் தகவலை கொடுக்க வேண்டுமா? ஆமாம் அல்லது இல்லை என்று சொல்லுங்கள். | అక్క, ఇంకో గర్భిణీ స్త్రీ వివరాలు ఇవ్వాలా? అవును లేదా కాదు అని చెప్పండి. |
| history_match | Mujhe {village} ki {name} ka purana record mila hai. Kya yeh wahi patient hain? Haan ya nahi batayein. | {village} ஊரைச் சேர்ந்த {name}-ன் பழைய பதிவு உள்ளது. இவர் அதே patient தானா? ஆமாம் அல்லது இல்லை என்று சொல்லுங்கள். | {village} గ్రామానికి చెందిన {name} పాత రికార్డు దొరికింది. వీరు అదే patient నా? అవును లేదా కాదు అని చెప్పండి. |
| history_yes | Theek hai. Kya aap unka pichhla visit summary sunna chahti hain? Haan ya nahi batayein. | சரி. அவர்களின் முந்தைய visit summary கேட்க வேண்டுமா? ஆமாம் அல்லது இல்லை என்று சொல்லுங்கள். | సరే. వారి పాత visit summary వినాలనుకుంటున్నారా? అవును లేదా కాదు అని చెప్పండి. |
| history_another | Theek hai. Ek aur purana record mila: {name}, gaon {village}, pati/guardian {spouse}. Kya yeh wahi patient hain? | சரி. இன்னொரு பழைய பதிவு உள்ளது: {name}, ஊர் {village}, கணவர்/பாதுகாவலர் {spouse}. இவர் அதே patient தானா? | సరే. ఇంకో పాత రికార్డు ఉంది: {name}, గ్రామం {village}, భర్త/సంరక్షకుడు {spouse}. వీరు అదే patient నా? |
| history_no_more | Theek hai, main ise alag patient ke roop mein darj karungi. | சரி, இதை புதிய patient-ஆக பதிவு செய்கிறேன். | సరే, దీనిని కొత్త patient గా నమోదు చేస్తాను. |
| read_history | Pichhle visit mein {name} ke liye {details} darj tha. Yeh sirf purana record hai. {prompt} | முந்தைய visit-ல் {name}-க்கு {details} பதிவாகியுள்ளது. இது பழைய பதிவு மட்டுமே. {prompt} | గత visit లో {name} కు {details} నమోదైంది. ఇది పాత రికార్డు మాత్రమే. {prompt} |
| read_history_default_prompt | Ab naye visit ki jaankari confirm karein. | இப்போது புதிய visit தகவலை confirm செய்யுங்கள். | ఇప్పుడు కొత్త visit వివరాలు confirm చేయండి. |
| history_repeat | Didi, kya aap pichhla visit summary sunna chahti hain? Haan ya nahi batayein. | அக்கா, முந்தைய visit summary கேட்க வேண்டுமா? ஆமாம் அல்லது இல்லை என்று சொல்லுங்கள். | అక్క, పాత visit summary వినాలనుకుంటున్నారా? అవును లేదా కాదు అని చెప్పండి. |
| correct_generic | Theek hai Didi. Kaunsi jaankari galat hai? Sahi value dobara batayein. | சரி அக்கா. எந்த தகவல் தவறாக உள்ளது? சரியான மதிப்பை மீண்டும் சொல்லுங்கள். | సరే అక్క. ఏ వివరం తప్పుగా ఉంది? సరైన విలువ మళ్ళీ చెప్పండి. |
| correct_spouse | Theek hai Didi. Pati ka sahi naam batayein. | சரி அக்கா. கணவரின் சரியான பெயரைச் சொல்லுங்கள். | సరే అక్క. భర్త సరైన పేరు చెప్పండి. |
| correct_patient | Theek hai Didi. Marij ka sahi naam batayein. | சரி அக்கா. Patient-ன் சரியான பெயரைச் சொல்லுங்கள். | సరే అక్క. Patient సరైన పేరు చెప్పండి. |
| correct_bp | Theek hai Didi. Sahi BP ke dono number batayein. | சரி அக்கா. சரியான BP இரண்டு எண்களையும் சொல்லுங்கள். | సరే అక్క. సరైన BP రెండు నంబర్లు చెప్పండి. |
| correct_weeks | Theek hai Didi. Pregnancy ke sahi hafte batayein. | சரி அக்கா. Pregnancy-யின் சரியான weeks எண்ணிக்கையைச் சொல்லுங்கள். | సరే అక్క. Pregnancy సరైన weeks చెప్పండి. |
| correct_hb | Theek hai Didi. Sahi Hemoglobin (Hb) batayein. | சரி அக்கா. சரியான Hemoglobin (Hb) அளவைச் சொல்லுங்கள். | సరే అక్క. సరైన Hemoglobin (Hb) చెప్పండి. |
| repeat_bp | Kripya BP ke dono number dobara batayein. | தயவுசெய்து BP இரண்டு எண்களையும் மீண்டும் சொல்லுங்கள். | దయచేసి BP రెండు నంబర్లు మళ్ళీ చెప్పండి. |
| repeat_generic | {message} Kripya sahi value dobara batayein. | {message} தயவுசெய்து சரியான மதிப்பை மீண்டும் சொல்லுங்கள். | {message} దయచేసి సరైన విలువ మళ్ళీ చెప్పండి. |
| repeat_escalate_prefix | Screening {level} hai, kripya PHC team se jaldi sampark karein. {prompt} | Screening {level} ஆக உள்ளது, உடனே PHC குழுவைத் தொடர்பு கொள்ளவும். {prompt} | Screening {level} గా ఉంది, వెంటనే PHC టీమ్ ని సంప్రదించండి. {prompt} |
| collect_confirm_fallback | Kripya in jaankariyon ko confirm karein. | தயவுசெய்து இந்த தகவல்களை confirm செய்யுங்கள். | దయచేసి ఈ వివరాలను confirm చేయండి. |
| readback_empty | abhi koi visit detail record nahi hui | இதுவரை எந்த visit தகவலும் பதிவாகவில்லை | ఇప్పటివరకు ఏ visit వివరాలు నమోదు కాలేదు |
| readback_changed | Theek hai, badal diya: {changed_summary}. Screening {level} hai. Kya yeh jaankari sahi hai? | சரி, மாற்றப்பட்டது: {changed_summary}. Screening {level} ஆக உள்ளது. இந்த தகவல் சரியா? | సరే, మార్చబడింది: {changed_summary}. Screening {level} గా ఉంది. ఈ వివరాలు సరైనవేనా? |
| readback | Didi, confirm karein: {summary}. Screening {level} hai. Kya yeh jaankari sahi hai? | அக்கா, confirm செய்யுங்கள்: {summary}. Screening {level} ஆக உள்ளது. இந்த தகவல் சரியா? | అక్క, confirm చేయండి: {summary}. Screening {level} గా ఉంది. ఈ వివరాలు సరైనవేనా? |
| language_switched | Theek hai Didi, ab main Hindi mein baat karungi. | சரி அக்கா, இனி தமிழில் பேசுகிறேன். | సరే అక్క, ఇక తెలుగులో మాట్లాడుతున్నాను. |
| language_menu | Bhasha badalne ke liye: Hindi ke liye 1, Tamil ke liye 2, Telugu ke liye 3 dabayein. | மொழி மாற்ற: Hindi-க்கு 1, Tamil-க்கு 2, Telugu-க்கு 3 அழுத்தவும். | భాష మార్చడానికి: Hindi కోసం 1, Tamil కోసం 2, Telugu కోసం 3 నొక్కండి. |

## Symptom Labels

| SymptomCode | Hindi (hi) | Tamil (	a) | Telugu (	e) |
|---|---|---|---|
| SEVERE_HEADACHE | tez sar dard | கடுமையான தலைவலி | తీవ్రమైన తలనొప్పి |
| BLURRED_VISION | dhundhla dikhna | மங்கலான பார்வை | మసక చూపు |
| EPIGASTRIC_PAIN | pet mein dard | வயிற்று வலி | కడుపు నొప్పి |
| CONVULSION | daura | வலிப்பு | ఫిట్స్ |
| UNCONSCIOUS | behoshi | மயக்கம் | స్పృహ కోల్పోవడం |
| VAGINAL_BLEEDING | bleeding | bleeding | bleeding |
| LABOUR_PAINS | prasav dard | பிரசவ வலி | పురిటి నొప్పులు |
| LEAKING_FLUID | paani aana | நீர் கசிவு | నీరు కారడం |
| REDUCED_FETAL_MOVEMENT | bachche ki halchal kam | குழந்தை அசைவு குறைவு | బిడ్డ కదలికలు తగ్గడం |
| SWELLING | soojan | வீக்கம் | వాపు |
