import os
import json
import re
import numpy as np
from dotenv import load_dotenv
from retriever import model as _embed_model

# -- Gemini setup --------------------------------------------------------------
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]
_gemini_client = None
_qa_cache = {}


def _get_gemini():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        from google.genai import types
        # 15s timeout and 1 attempt for snappy responsiveness without premature timeouts
        _gemini_client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(
                timeout=15000,
                retry_options=types.HttpRetryOptions(attempts=1)
            )
        )
        return _gemini_client
    except Exception as e:
        print(f"[Gemini] init error: {e}")
        return None


# -- Sentence splitting --------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    """Split text into clean sentences, preserving key-value lines and legal references."""
    if not text or not text.strip():
        return []

    abbrevs = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Rs.", "No.", "Sec.",
        "Inc.", "Ltd.", "Co.", "Jan.", "Feb.", "Mar.", "Apr.",
        "Aug.", "Sept.", "Oct.", "Nov.", "Dec.", "Adv.", "vs."
    ]

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    merged, current = [], ""
    for line in lines:
        if not current:
            current = line
        elif current[-1] in ".!?:" or ":" in line:
            merged.append(current)
            current = line
        else:
            current += " " + line
    if current:
        merged.append(current)

    sentences = []
    for unit in merged:
        unit = re.sub(r"\s+", " ", unit).strip()
        protected = unit
        for abbr in abbrevs:
            protected = re.sub(r"\b" + re.escape(abbr), abbr[:-1] + "___", protected)
        for part in re.split(r"(?<=[.!?])\s+", protected):
            restored = part.replace("___", ".").strip()
            if len(restored) > 2:
                sentences.append(restored)

    return sentences or [re.sub(r"\s+", " ", text).strip()]


# -- Language Configurations ---------------------------------------------------
LANGUAGE_CONFIG = {
    "en": {
        "name": "English",
        "desc": "plain, direct, everyday English (1-3 sentences)",
        "not_found": "This information is not mentioned in the uploaded document.",
        "example": "The monthly rent is Rs. 15,000 payable on or before the 5th of each month."
    },
    "hi": {
        "name": "Hindi (हिंदी)",
        "desc": "clear, natural Hindi in Devanagari script (हिंदी में 1-3 वाक्य)",
        "not_found": "यह जानकारी अपलोड किए गए दस्तावेज़ में मौजूद नहीं है।",
        "example": "मासिक किराया ₹15,000 है जो प्रत्येक माह की 5 तारीख तक देय है।"
    },
    "mr": {
        "name": "Marathi (मराठी)",
        "desc": "clear, natural Marathi in Devanagari script (मराठीत 1-3 वाक्ये)",
        "not_found": "ही माहिती अपलोड केलेल्या दस्तऐवजात नमूद केलेली नाही.",
        "example": "दरमहा भाडे ₹15,000 असून ते प्रत्येक महिन्याच्या 5 तारखेपूर्वी देणे आवश्यक आहे."
    }
}


# -- Auto Language Detection ---------------------------------------------------
def detect_question_language(question: str, fallback_lang: str = "en") -> str:
    """
    Auto-detect whether the user asked in Marathi, Hindi, or English.
    Handles both Devanagari script markers and transliterated Roman script (e.g. kiti, kitna).
    """
    if not question or not question.strip():
        return fallback_lang

    q = question.strip()
    q_lower = q.lower()

    # Marathi markers (Devanagari + transliteration)
    marathi_devanagari = ["आहे", "नाही", "किती", "काय", "कोणाचे", "कोणाला", "कधी", "आहेत", "दिले", "केले", "करावे", "भाडे", "घरमालक", "भाडेकरू", "थकबाकी", "पावती"]
    marathi_roman = ["kiti", "aahe", "ahet", "nahi", "konache", "kadhi", "bhade", "malkache", "bhadikaru", "kay aahe", "ghar malak"]

    if any(m in q for m in marathi_devanagari) or any(re.search(r"\b" + re.escape(r) + r"\b", q_lower) for r in marathi_roman):
        return "mr"

    # Hindi markers (Devanagari + transliteration)
    hindi_devanagari = ["है", "हैं", "कितना", "कितनी", "क्या", "किसका", "किसने", "कब", "कहाँ", "किराया", "मकान मालिक", "किरायेदार", "बकाया", "समझौता"]
    hindi_roman = ["kitna", "kitni", "kaun", "kya", "kab", "kisko", "kiska", "hai", "hain", "kiraya", "karega", "hoga", "makan malik"]

    if any(m in q for m in hindi_devanagari) or any(re.search(r"\b" + re.escape(r) + r"\b", q_lower) for r in hindi_roman):
        return "hi"

    # If question contains Devanagari characters, it is never English
    devanagari_count = len(re.findall(r"[\u0900-\u097F]", q))
    if devanagari_count >= 2:
        if fallback_lang in ["mr", "hi"]:
            return fallback_lang
        # Check common Marathi letters or grammar (ळ, ण, च, etc.)
        if "ळ" in q or any(w in q for w in ["चे", "च्या", "ला", "ने", "वर"]):
            return "mr"
        return "hi"

    # If fallback is explicitly provided (e.g. user toggled Marathi or Hindi in the UI)
    if fallback_lang in ["hi", "mr"]:
        return fallback_lang

    return "en"


# -- Semantic similarity -------------------------------------------------------
def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    emb = _embed_model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


# -- Gemini Grounded Q&A -------------------------------------------------------
def _ask_gemini(question: str, context: str, best_page: int | None = None, lang: str = "en") -> dict | None:
    client = _get_gemini()
    if not client:
        return None

    # Detect the actual language of the question (handles Hindi, Marathi, English)
    effective_lang = detect_question_language(question, fallback_lang=lang)
    cfg = LANGUAGE_CONFIG.get(effective_lang, LANGUAGE_CONFIG["en"])
    lang_name = cfg["name"]
    lang_desc = cfg["desc"]
    not_found_msg = cfg["not_found"]
    example_ans = cfg["example"]

    prompt = f"""You are LexiGuide, an expert AI legal document assistant supporting English, Hindi, and Marathi.
Your task is to accurately answer the user's question using ONLY the provided legal document text.

DOCUMENT EXCERPTS (Note: The legal document is written in English):
{context}

USER QUESTION:
{question}

DETECTED TARGET ANSWER LANGUAGE: {lang_name}

MANDATORY RULES & INSTRUCTIONS:
1. Strict Grounding: Answer strictly using facts present in the DOCUMENT EXCERPTS. Do NOT hallucinate. Note: In rent receipts and payment vouchers, 'Received from' identifies the tenant or payer, and the person issuing/signing is the landlord or payee.
2. CRITICAL CROSS-LINGUAL RULE (ENGLISH PDF -> USER LANGUAGE):
   - EVEN THOUGH THE UPLOADED LEGAL DOCUMENT IS IN ENGLISH, YOUR ANSWER EXPLANATION MUST BE IN {lang_name} (the language the user asked in)!
   - If the user asks in Marathi (e.g. 'भाडे किती आहे?' or 'घरमालक कोण आहे?' or 'bhade kiti aahe'): You MUST explain and formulate the answer strictly in clear, natural Marathi (मराठी in Devanagari script).
   - If the user asks in Hindi (e.g. 'किराया कितना है?' or 'landlord kaun hai?' or 'rent kitna hai?'): You MUST explain and formulate the answer strictly in clear, natural Hindi (हिंदी in Devanagari script).
   - If the user asks in English: Provide the answer in English.
   - Do NOT reply in English if the question was asked in Hindi or Marathi!
   - Accurately translate and convey all names, amounts, dates, and legal facts into {lang_name}.
   - Example style for {lang_name}: "{example_ans}"
3. Source Clause Citation:
   - In "clause", quote the EXACT original English excerpt or sentence from the document as proof of citation.
4. If the answer is present in the document:
   - Provide a clear, direct, and well-explained answer in {lang_name} (1-3 sentences).
   - Identify the exact page number where this information is found.
   - Set "found" to true and provide an accurate confidence (e.g. 0.85 to 0.99).
5. If the answer is NOT present or cannot be determined:
   - Set "found" to false.
   - Answer: "{not_found_msg}"
   - Set "page" to null, "clause" to null, and "confidence" to 0.0.

Respond ONLY with valid JSON in this exact structure:
{{
  "found": true,
  "answer": "Direct answer in {lang_name}",
  "page": 1,
  "clause": "exact quote from document in English",
  "confidence": 0.95
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
            # Clean possible markdown wrap
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            if isinstance(data, dict) and "answer" in data:
                return data
        except Exception as e:
            print(f"[Gemini] model {model} failed: {e}")
            continue

    return None


def _format_local_answer(best_sent: str, question: str, lang: str, page: int) -> str:
    if lang == "en":
        return best_sent

    amounts = re.findall(r"(?:Rs\.?|INR|₹)\s*[\d,]+(?:\.\d+)?|\b\d+,\d+\b", best_sent, re.IGNORECASE)
    amount_str = amounts[0] if amounts else ""
    dates = re.findall(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", best_sent, re.IGNORECASE)
    date_str = dates[0] if dates else ""
    
    # Extract payer/tenant e.g. "Received from Mr. Harshal Patond"
    payer_match = re.search(r"(?:received from|paid by|tenant|payer)\s*[:\-]?\s*([A-Za-z\.\s]{3,40})", best_sent, re.IGNORECASE)
    payer_name = payer_match.group(1).strip(" .,[]") if payer_match else ""

    q_lower = question.lower()
    
    # 1. Rent / Amount questions
    if any(w in q_lower for w in ["rent", "amount", "bhade", "भाडे", "रक्कम", "किराया", "राशि", "kitna", "kiti"]):
        if amount_str:
            if lang == "mr":
                return f"दस्तऐवजानुसार, देय भाडे किंवा रक्कम {amount_str} आहे."
            elif lang == "hi":
                return f"दस्तावेज़ के अनुसार, देय किराया या कुल राशि {amount_str} है।"

    # 2. Tenant / Payer questions
    if any(w in q_lower for w in ["tenant", "payer", "bhadekaru", "भाडेकरू", "किरायेदार", "bhadut", "konache", "kaun", "koni"]):
        if payer_name:
            if lang == "mr":
                return f"दस्तऐवजानुसार, भाडेकरूचे नाव {payer_name} आहे."
            elif lang == "hi":
                return f"दस्तावेज़ के अनुसार, किरायेदार का नाम {payer_name} है।"

    # 3. Landlord questions
    if any(w in q_lower for w in ["landlord", "owner", "ghar malak", "घरमालक", "मकान मालिक"]):
        landlord_match = re.search(r"(?:landlord|owner|lessor|signature of landlord)\s*[:\-]?\s*([A-Za-z\.\s]{3,40})", best_sent, re.IGNORECASE)
        landlord_name = landlord_match.group(1).strip(" .,[]") if landlord_match else ""
        if landlord_name:
            if lang == "mr":
                return f"दस्तऐवजानुसार, घरमालकाचे नाव {landlord_name} आहे."
            elif lang == "hi":
                return f"दस्तावेज़ के अनुसार, मकान मालिक का नाम {landlord_name} है।"

    # 4. Date questions
    if any(w in q_lower for w in ["date", "kadhi", "कधी", "तारीख", "kab", "कब", "दिनांक"]):
        if date_str:
            if lang == "mr":
                return f"दस्तऐवजातील नोंदणीकृत तारीख {date_str} आहे."
            elif lang == "hi":
                return f"दस्तावेज़ में उल्लिखित तारीख {date_str} है।"

    # 5. Arrears / Dues
    if any(w in q_lower for w in ["arrear", "due", "penalty", "दंड", "थकबाकी", "बकाया"]):
        if any(w in best_sent.lower() for w in ["nil", "no", "zero", "0"]):
            if lang == "mr":
                return "दस्तऐवजात कोणतीही जुनी थकबाकी किंवा दंड नाही (निरंक)."
            elif lang == "hi":
                return "दस्तावेज़ में कोई पुराना बकाया या जुर्माना नहीं है (शून्य)।"

    # General sentence representation in target language
    clean_sent = re.sub(r"[\[\]]", "", best_sent).strip()
    if lang == "mr":
        return f"दस्तऐवजातील नोंदीनुसार (पान क्र. {page}): {clean_sent}"
    elif lang == "hi":
        return f"दस्तावेज़ के विवरणानुसार (पृष्ठ {page}): {clean_sent}"
    return clean_sent


# -- Strict Local Fallback (Used if Gemini is unavailable) ----------------------
def _strict_local_fallback(question: str, ranked: list, lang: str = "en") -> dict:
    cfg = LANGUAGE_CONFIG.get(lang.lower(), LANGUAGE_CONFIG["en"])
    no_answer = {
        "answer": cfg["not_found"],
        "page": None, "clause": None, "confidence": 0.0, "lang": lang
    }
    if not ranked:
        return no_answer

    # Tokenize question words (ignoring common stopwords)
    stops = {
        "what", "is", "the", "a", "an", "in", "of", "to", "for", "and", "or",
        "who", "when", "where", "how", "are", "do", "does", "did", "this",
        "that", "it", "be", "by", "on", "at", "from", "with", "as", "about",
        "kya", "hai", "kaun", "kab", "kitna", "kiti", "aahe", "kay", "kon"
    }
    q_words = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 2 and w.lower() not in stops]

    # Gather unique candidate sentences
    candidates = []
    seen = set()
    for r in ranked:
        p_num = r.get("page", 1)
        for sent in split_sentences(r.get("text", "")):
            s_clean = sent.strip()
            if s_clean and len(s_clean) > 10 and s_clean not in seen:
                seen.add(s_clean)
                candidates.append((s_clean, p_num))

    if not candidates:
        return no_answer

    # Batch encode all candidates at once for instant CPU performance (0.02s)
    all_texts = [c[0] for c in candidates]
    try:
        embs = _embed_model.encode([question] + all_texts, convert_to_numpy=True, normalize_embeddings=True)
        q_emb = embs[0]
        sent_embs = embs[1:]
        scores = np.dot(sent_embs, q_emb)
    except Exception as e:
        print(f"[Vector Fallback Error]: {e}")
        return no_answer

    valid = []
    for idx, (sent, p_num) in enumerate(candidates):
        s_words = set(w.lower() for w in re.findall(r"\w+", sent))
        overlap = any(qw in s_words for qw in q_words) if q_words else True
        if overlap and scores[idx] >= 0.20:
            valid.append((float(scores[idx]), sent, p_num))

    if valid:
        valid.sort(key=lambda x: x[0], reverse=True)
        best_score, best_sent, best_page = valid[0]
        formatted_answer = _format_local_answer(best_sent, question, lang, best_page)
        return {
            "answer": formatted_answer,
            "page": best_page,
            "clause": best_sent,
            "confidence": round(best_score, 3),
            "lang": lang
        }

    return no_answer


# -- Public API ----------------------------------------------------------------
def extract_answer(question: str, results: list, pages: list | None = None, lang: str = "en") -> dict:
    """
    Main entry point for Q&A supporting English, Hindi, and Marathi.
    Returns: {"answer": str, "page": int|None, "clause": str|None, "confidence": float, "lang": str}
    """
    effective_lang = detect_question_language(question, fallback_lang=lang)
    cfg = LANGUAGE_CONFIG.get(effective_lang, LANGUAGE_CONFIG["en"])
    no_answer = {
        "answer": cfg["not_found"],
        "page": None, "clause": None, "confidence": 0.0,
        "lang": effective_lang
    }

    if not question or not question.strip():
        empty_prompts = {
            "en": "Please enter a question.",
            "hi": "कृपया कोई प्रश्न दर्ज करें।",
            "mr": "कृपया एक प्रश्न प्रविष्ट करा."
        }
        return {"answer": empty_prompts.get(effective_lang, "Please enter a question."), "page": None, "clause": None, "confidence": 0.0, "lang": effective_lang}

    cache_key = (question.strip().lower(), effective_lang)
    if cache_key in _qa_cache:
        return _qa_cache[cache_key]

    # 1. Build compact, high-precision context
    context_parts = []
    best_page = 1

    if results:
        # Use top 4 most semantically relevant chunks for fast token processing
        seen = set()
        for r in results[:5]:
            txt = r.get("text", "").strip()
            if txt and txt not in seen:
                seen.add(txt)
                context_parts.append(f"[Page {r.get('page', 1)}]:\n{txt}")
        best_page = results[0].get("page", 1)
    elif pages:
        # Fallback to first few pages if no search results
        for p in pages[:3]:
            context_parts.append(f"[Page {p['page']}]:\n{p['text']}")
        best_page = pages[0]["page"]
    else:
        return no_answer

    context = "\n\n".join(context_parts)

    # 2. Query Gemini
    gemini_res = _ask_gemini(question, context, best_page=best_page, lang=effective_lang)
    if gemini_res is not None:
        if gemini_res.get("found", True) and gemini_res.get("answer"):
            page_val = gemini_res.get("page")
            if page_val is None or str(page_val).lower() == "null":
                page_val = best_page
            try:
                page_val = int(page_val)
            except (ValueError, TypeError):
                page_val = best_page

            clause_val = gemini_res.get("clause")
            if not clause_val or str(clause_val).lower() == "null":
                clause_val = context_parts[0] if context_parts else None

            conf_val = float(gemini_res.get("confidence", 0.90))

            res = {
                "answer": gemini_res["answer"].strip(),
                "page": page_val,
                "clause": clause_val,
                "confidence": round(conf_val, 3),
                "lang": effective_lang
            }
            _qa_cache[cache_key] = res
            return res
        else:
            # Gemini explicitly confirmed the answer is NOT in the document
            _qa_cache[cache_key] = no_answer
            return no_answer

    # 3. Fallback: Fast vector retrieval ONLY if Gemini was completely unreachable
    ranked = []
    for r in (results or []):
        t = r.get("text", "").strip()
        if t:
            ranked.append({
                "text": t,
                "page": r.get("page", 1),
                "score": r.get("score", 0.0)
            })
    ranked.sort(key=lambda x: x["score"], reverse=True)
    ans = _strict_local_fallback(question, ranked, lang=effective_lang)
    ans["lang"] = effective_lang
    _qa_cache[cache_key] = ans
    return ans
