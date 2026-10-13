import os
import re
import json
import hashlib
from qa_engine import _get_gemini, GEMINI_MODELS

# In-memory cache for fast repeated risk queries
_risk_cache = {}


def _hash_doc(text: str, lang: str) -> str:
    return hashlib.md5(f"{lang}:{text[:1000]}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# LOCAL HEURISTIC RISK SCANNER (High-precision fallback engine)
# ---------------------------------------------------------------------------
HEURISTIC_RULES = [
    {
        "id": "sec_deposit_forfeit",
        "severity": "high",
        "pattern": r"(?:deposit.*(?:forfeit|non-refundable|not refundable|retained)|forfeiture of security deposit)",
        "title_en": "Security Deposit Forfeiture / Non-Refundable Clause",
        "title_hi": "सुरक्षा जमा (Security Deposit) जब्ती का जोखिम",
        "title_mr": "सुरक्षा ठेव (Security Deposit) जप्तीचा धोका",
        "exp_en": "The clause indicates that the security deposit may be forfeited or is non-refundable, posing a financial risk of loss.",
        "exp_hi": "इस शर्त के अनुसार आपकी सुरक्षा जमा राशि जब्त हो सकती है या वापस नहीं मिलेगी, जिससे वित्तीय नुकसान का खतरा है।",
        "exp_mr": "या अटीनुसार सुरक्षा ठेव जप्त केली जाऊ शकते किंवा परत न मिळण्याचा धोका आहे, ज्यामुळे आर्थिक नुकसान होऊ शकते.",
        "rec_en": "Request modifying this clause: 'Security deposit must be refunded within 7 days of handover, subject only to actual substantiated damages.'",
        "rec_hi": "शर्त में बदलाव मांगें: 'कब्जा सौंपने के 7 दिनों के भीतर सुरक्षा जमा वापस किया जाए, केवल वास्तविक क्षति ही काटी जा सके।'",
        "rec_mr": "बदल सुचवा: 'जागा सोडल्यापासून 7 दिवसांच्या आत प्रत्यक्ष नुकसानीची रक्कम वगळून संपूर्ण ठेव परत केली जावी.'"
    },
    {
        "id": "unilateral_term",
        "severity": "high",
        "pattern": r"(?:terminate.*(?:without (?:any )?notice|immediately|forthwith)|right to terminate without notice|evict.*without)",
        "title_en": "Unilateral Immediate Termination Without Notice",
        "title_hi": "बिना नोटिस एकतरफा अनुबंध समाप्ति का अधिकार",
        "title_mr": "विना सूचना एकतर्फी करार समाप्तीचा अधिकार",
        "exp_en": "The agreement allows immediate termination or eviction without reasonable advance notice period.",
        "exp_hi": "यह शर्त दूसरे पक्ष को बिना किसी अग्रिम नोटिस या समय दिए तुरंत अनुबंध समाप्त करने या बेदखल करने की अनुमति देती है।",
        "exp_mr": "ही अट दुसऱ्या पक्षाला कोणतीही पूर्वसूचना न देता तात्काळ करार संपवण्याची किंवा निष्कासित करण्याची मुभा देते.",
        "rec_en": "Require a mandatory minimum 30-day written cure notice period prior to any termination action.",
        "rec_hi": "कम से कम 30 दिनों का अनिवार्य लिखित नोटिस पीरियड जोड़ने का अनुरोध करें।",
        "rec_mr": "कोणत्याही कारवाईपूर्वी किमान 30 दिवसांची लेखी नोटीस देण्याची अट समाविष्ट करा."
    },
    {
        "id": "lock_in_penalty",
        "severity": "medium",
        "pattern": r"(?:lock-in.*(?:period|month)|lock in period|shall not vacate.*prior|entire remaining rent)",
        "title_en": "Strict Lock-in Period & Exit Penalty",
        "title_hi": "कड़ा लॉक-इन पीरियड और बाहर निकलने पर जुर्माना",
        "title_mr": "कडक लॉक-इन कालावधी आणि दंड",
        "exp_en": "A lock-in period prevents ending the contract early without paying compensation or forfeiting deposits.",
        "exp_hi": "लॉक-इन अवधि के कारण आप समय से पहले अनुबंध समाप्त नहीं कर सकते; ऐसा करने पर पूरा किराया या जुर्माना देना पड़ सकता है।",
        "exp_mr": "लॉक-इन कालावधीमुळे मुदतीपूर्वी करार सोडल्यास दंड किंवा उर्वरित भाडे भरावे लागण्याचा धोका असतो.",
        "rec_en": "Negotiate an emergency exit clause permitting early termination for job transfer or unforeseen emergencies with 1 month notice.",
        "rec_hi": "नौकरी तबादले या आपात स्थिति में 1 माह के नोटिस पर समय से पूर्व खाली करने की छूट मांगें।",
        "rec_mr": "बदली किंवा आणीबाणीच्या परिस्थितीत 1 महिन्याची नोटीस देऊन करार संपवण्याची मुभा मागा."
    },
    {
        "id": "unilateral_rent_hike",
        "severity": "medium",
        "pattern": r"(?:increas.*rent.*(?:discretion|sole option|any time)|escalat.*rent.*without)",
        "title_en": "Unilateral Rent Escalation Clause",
        "title_hi": "मनमाने ढंग से किराया बढ़ाने की एकतरफा शर्त",
        "title_mr": "एकतर्फी भाडेवाढीची अट",
        "exp_en": "Permits arbitrary rent increments at the landlord's discretion without a predefined cap or mutual agreement.",
        "exp_hi": "यह शर्त मकान मालिक को बिना आपकी सहमति के अपनी मर्जी से कभी भी किराया बढ़ाने की छूट देती है।",
        "exp_mr": "ही अट संमतीशिवाय किंवा मर्यादेशिवाय परस्पर भाडे वाढवण्याचा एकतर्फी अधिकार देते.",
        "rec_en": "Cap rent escalation to a maximum of 5%–10% annually upon mutual renewal only.",
        "rec_hi": "किराया वृद्धि को अधिकतम 5%–10% तक सीमित करने और केवल पारस्परिक नवीनीकरण पर लागू करने की शर्त रखें।",
        "rec_mr": "वार्षिक भाडेवाढ 5% ते 10% पर्यंत मर्यादित ठेवा आणि ती परस्पर संमतीनेच लागू असावी."
    },
    {
        "id": "high_interest_penalty",
        "severity": "medium",
        "pattern": r"(?:interest.*(?:1[89]%|[2-9]\d%|per day|per week)|late fee.*(?:1[89]%|[2-9]\d%))",
        "title_en": "Excessive Late Payment Penalty / Interest Rate",
        "title_hi": "अत्यधिक लेट फीस या भारी ब्याज दर",
        "title_mr": "अतिउंच विलंब शुल्क किंवा व्याजदर",
        "exp_en": "Penal interest exceeding standard commercial rates (18%+ per annum or aggressive daily penalties).",
        "exp_hi": "विलंब होने पर 18% से अधिक ब्याज या दैनिक भारी जुर्माना लागू हो सकता है जो कानूनी रूप से अनुचित है।",
        "exp_mr": "उशिरा देयकावर 18% पेक्षा जास्त व्याज किंवा दैनंदिन मोठा दंड आकारला जाऊ शकतो.",
        "rec_en": "Cap late interest to standard 1% to 1.5% per month with a reasonable 5-day grace period.",
        "rec_hi": "विलंब शुल्क को 1% से 1.5% प्रति माह तक सीमित करें और 5 दिनों का ग्रेस पीरियड मांगें।",
        "rec_mr": "विलंब व्याज दरमहा जास्तीत जास्त 1% ते 1.5% पर्यंत मर्यादित ठेवा आणि 5 दिवसांची सवलत मागा."
    },
    {
        "id": "unrestricted_inspection",
        "severity": "low",
        "pattern": r"(?:enter.*(?:any time|without notice|at all times)|inspect.*without.*(?:notice|permission))",
        "title_en": "Unrestricted Entry & Privacy Intrusion",
        "title_hi": "बिना पूर्व सूचना प्रवेश और निजता का हनन",
        "title_mr": "पूर्वसूचनेशिवाय प्रवेश आणि गोपनीयतेचा भंग",
        "exp_en": "Allows entry into premises at any time without mandatory advance notice or consent.",
        "exp_hi": "परिसर में बिना पूर्व सूचना या अनुमति के किसी भी समय प्रवेश करने की अनुमति देता है।",
        "exp_mr": "जागेमध्ये पूर्वसूचनेशिवाय किंवा परवानगीशिवाय कधीही प्रवेश करण्याची मुभा देते.",
        "rec_en": "Specify that inspections require minimum 24 hours prior written or telephonic notice during reasonable daytime hours.",
        "rec_hi": "शर्त रखें कि निरीक्षण केवल उचित दिन के समय में कम से कम 24 घंटे पहले सूचना देकर ही किया जाएगा।",
        "rec_mr": "तपासणीसाठी दिवसाच्या वेळेत किमान 24 तास आधी पूर्वसूचना देणे बंधनकारक करा."
    },
    {
        "id": "broad_indemnity",
        "severity": "medium",
        "pattern": r"(?:indemnif.*(?:harmless|against all|all losses|any damage)|solely liable.*any damage)",
        "title_en": "Overly Broad Indemnification & Liability Shift",
        "title_hi": "अत्यधिक क्षतिपूर्ति और संपूर्ण दायित्व लादना",
        "title_mr": "अमर्याद नुकसानभरपाई आणि दायित्व",
        "exp_en": "Shifts liability for structural repairs, force majeure, or third-party issues onto the tenant/user.",
        "exp_hi": "प्राकृतिक आपदा या ढांचागत खराबी का हर्जाना भी आपके ऊपर डालने की कोशिश की गई है।",
        "exp_mr": "नैसर्गिक आपत्ती किंवा इमारतीच्या स्ट्रक्चरल नुकसानीची जबाबदारीही तुमच्यावर टाकण्याचा प्रयत्न.",
        "rec_en": "Clarify that structural repairs and normal wear-and-tear remain the exclusive responsibility of the owner.",
        "rec_hi": "स्पष्ट करें कि प्राकृतिक घिसाव और इमारत की बड़ी मरम्मत केवल मालिक की जिम्मेदारी होगी।",
        "rec_mr": "स्पष्ट करा की इमारतीची मोठी दुरुस्ती आणि नैसर्गिक झीज ही केवळ मालकाची जबाबदारी असेल."
    }
]


def _run_heuristic_risk_analysis(pages: list[dict], lang: str = "en") -> dict:
    """Fast, accurate rule-based scanner that inspects every page for legal traps."""
    flags = []
    seen_ids = set()
    total_score = 10  # Base safe score

    for page_obj in pages:
        p_num = page_obj.get("page", 1)
        p_text = page_obj.get("text", "")

        for rule in HEURISTIC_RULES:
            if rule["id"] in seen_ids:
                continue
            match = re.search(rule["pattern"], p_text, re.IGNORECASE)
            if match:
                seen_ids.add(rule["id"])
                # Extract snippet around match
                start = max(0, match.start() - 60)
                end = min(len(p_text), match.end() + 140)
                snippet = p_text[start:end].strip()
                if start > 0:
                    snippet = "..." + snippet
                if end < len(p_text):
                    snippet = snippet + "..."

                sev = rule["severity"]
                if sev == "high":
                    total_score += 25
                elif sev == "medium":
                    total_score += 15
                else:
                    total_score += 8

                title_key = f"title_{lang}" if f"title_{lang}" in rule else "title_en"
                exp_key = f"exp_{lang}" if f"exp_{lang}" in rule else "exp_en"
                rec_key = f"rec_{lang}" if f"rec_{lang}" in rule else "rec_en"

                flags.append({
                    "id": rule["id"],
                    "severity": sev,
                    "title": rule[title_key],
                    "page": p_num,
                    "clause": snippet,
                    "explanation": rule[exp_key],
                    "recommendation": rule[rec_key]
                })

    # Clamp score
    final_score = min(95, max(12, total_score)) if flags else 12

    # If document has no high risks (e.g. rent receipt)
    if not flags:
        final_score = 10
        if lang == "hi":
            summary = "इस दस्तावेज़ में कोई असामान्य या खतरनाक कानूनी शर्तें (Red Flags) नहीं पाई गईं। यह एक सुरक्षित और मानक दस्तावेज़ प्रतीत होता है।"
        elif lang == "mr":
            summary = "या दस्तऐवजात कोणत्याही धोकादायक किंवा अवाजवी कायदेशीर अटी (Red Flags) आढळल्या नाहीत. हा एक सुरक्षित आणि प्रमाणित दस्तऐवज वाटतो."
        else:
            summary = "No alarming legal risks or biased clauses were detected. This appears to be a safe, standard document with low liability."
    else:
        high_cnt = sum(1 for f in flags if f["severity"] == "high")
        med_cnt = sum(1 for f in flags if f["severity"] == "medium")
        if lang == "hi":
            summary = f"दस्तावेज़ में {len(flags)} संभावित जोखिम बिंदु मिले ({high_cnt} उच्च, {med_cnt} मध्यम)। हस्ताक्षर करने से पहले चिह्नित शर्तों की समीक्षा करें।"
        elif lang == "mr":
            summary = f"दस्तऐवजात {len(flags)} संभाव्य जोखीम मुद्दे आढळले ({high_cnt} उच्च, {med_cnt} मध्यम). स्वाक्षरी करण्यापूर्वी या अटी तपासा."
        else:
            summary = f"Identified {len(flags)} potential risk points ({high_cnt} high severity, {med_cnt} medium). Review highlighted clauses before signing."

    risk_level = "High" if final_score >= 65 else ("Moderate" if final_score >= 35 else "Low")

    return {
        "score": final_score,
        "risk_level": risk_level,
        "summary": summary,
        "flags": flags,
        "stats": {
            "total": len(flags),
            "high": sum(1 for f in flags if f["severity"] == "high"),
            "medium": sum(1 for f in flags if f["severity"] == "medium"),
            "low": sum(1 for f in flags if f["severity"] == "low")
        }
    }


# ---------------------------------------------------------------------------
# GEMINI GENERATIVE RISK ANALYZER (Deep semantic audit)
# ---------------------------------------------------------------------------
def analyze_document_risks(pages: list[dict], chunks: list[dict] = None, lang: str = "en") -> dict:
    """
    Performs comprehensive Legal Risk & Red Flag analysis on the document.
    Uses Gemini for deep semantic auditing and falls back gracefully to
    the high-precision heuristic scanner.
    """
    if not pages:
        return _run_heuristic_risk_analysis([], lang)

    combined_text = "\n\n".join([f"--- PAGE {p.get('page', i+1)} ---\n" + p.get("text", "")[:2500] for i, p in enumerate(pages[:6])])
    cache_key = _hash_doc(combined_text, lang)
    if cache_key in _risk_cache:
        return _risk_cache[cache_key]

    client = _get_gemini()
    target_lang = (lang or "en").lower()
    lang_label = "Hindi (हिंदी)" if target_lang == "hi" else ("Marathi (मराठी)" if target_lang == "mr" else "Plain English")

    if client:
        prompt = f"""You are LexiGuide Legal Risk & Red Flag Auditor.
Analyze the following legal document text from the viewpoint of the signing party (e.g., tenant, buyer, employee, or consumer).
Identify unfair, one-sided, ambiguous, or risky legal clauses (e.g. deposit forfeiture, unilateral termination, unreasonable penalty, rent escalation, excessive liability, unfair inspection).

TARGET LANGUAGE FOR EXPLANATIONS & RECOMMENDATIONS: {lang_label}

DOCUMENT TEXT:
\"\"\"{combined_text[:9000]}\"\"\"

Generate JSON adhering strictly to this schema:
{{
  "score": <integer from 0 to 100 representing overall riskiness, where 0-25 is Low/Safe, 26-65 is Moderate Caution, 66-100 is High Risk>,
  "risk_level": "<'Low' | 'Moderate' | 'High'>",
  "summary": "<2-sentence executive summary of the document's risk posture in {lang_label}>",
  "flags": [
    {{
      "id": "<short_id_e.g._deposit_risk>",
      "severity": "<'high' | 'medium' | 'low'>",
      "title": "<Concise risk title in {lang_label}>",
      "page": <page number integer>,
      "clause": "<exact quote from document in English>",
      "explanation": "<Plain-language explanation in {lang_label} of why this clause is risky or unfair>",
      "recommendation": "<Actionable renegotiation proposal or counter-clause in {lang_label}>"
    }}
  ]
}}
"""
        for model in GEMINI_MODELS[:2]:
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=dict(response_mime_type="application/json")
                )
                raw = resp.text.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-z]*\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                data = json.loads(raw)
                if isinstance(data, dict) and "score" in data and "flags" in data:
                    flags = data.get("flags", [])
                    data["stats"] = {
                        "total": len(flags),
                        "high": sum(1 for f in flags if f.get("severity") == "high"),
                        "medium": sum(1 for f in flags if f.get("severity") == "medium"),
                        "low": sum(1 for f in flags if f.get("severity") == "low")
                    }
                    _risk_cache[cache_key] = data
                    return data
            except Exception as e:
                print(f"[RiskAnalyzer] Gemini model {model} error: {e}")
                continue

    # Fallback to local heuristic analyzer
    result = _run_heuristic_risk_analysis(pages, lang=target_lang)
    _risk_cache[cache_key] = result
    return result
