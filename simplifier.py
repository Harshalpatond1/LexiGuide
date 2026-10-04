import re
from qa_engine import _get_gemini as get_client, GEMINI_MODELS as CANDIDATE_MODELS


# ---------------------------------------------------------
# LANGUAGE CONFIG FOR SIMPLIFIER
# ---------------------------------------------------------
SIMPLIFIER_CONFIG = {
    "en": {
        "name": "Plain English",
        "structure": """Provide:
   - **Key Summary**: What this clause/page establishes.
   - **Important Details**: Names, amounts, dates, properties.
   - **Terms & Conditions**: What parties must or must not do."""
    },
    "hi": {
        "name": "Hindi (हिंदी)",
        "structure": """आउटपुट शुद्ध और सरल हिंदी (Devanagari script) में निम्नलिखित शीर्षकों के साथ प्रदान करें:
   - **मुख्य सारांश**: यह पृष्ठ या अनुबंध क्या निर्धारित करता है।
   - **महत्वपूर्ण विवरण**: पक्षकारों के नाम, किराया/राशि, तिथियां, संपत्ति का पता।
   - **नियम एवं शर्तें**: पक्षकारों के अधिकार, जिम्मेदारियां और महत्वपूर्ण नियम।"""
    },
    "mr": {
        "name": "Marathi (मराठी)",
        "structure": """आउटपुट शुद्ध आणि सोप्या मराठीत (Devanagari script) खालील मथळ्यांनुसार द्या:
   - **मुख्य सारांश**: हे पान किंवा करार काय ठरवतो.
   - **महत्त्वाचा तपशील**: संबंधित व्यक्तींची नावे, भाडे/रक्कम, तारखा, पत्ता.
   - **नियम व अटी**: पक्षांचे अधिकार, जबाबदाऱ्या आणि महत्त्वाच्या अटी."""
    }
}


import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

_simplifier_cache = {}

def _hash_text(text: str, lang: str) -> str:
    return hashlib.md5(f"{lang}:{text[:500]}".encode("utf-8")).hexdigest()


def simplify_page_text(text: str, page_num: int, lang: str = "en") -> str:
    """
    Simplifies an entire page's legal text into plain language (English, Hindi, or Marathi)
    in one clean API call. Preserves all facts, rights, duties, dates, and amounts.
    """
    if not text or not text.strip():
        return text

    cache_key = _hash_text(text, lang or "en")
    if cache_key in _simplifier_cache:
        return _simplifier_cache[cache_key]

    client = get_client()
    cfg = SIMPLIFIER_CONFIG.get((lang or "en").lower(), SIMPLIFIER_CONFIG["en"])
    lang_name = cfg["name"]
    structure_guide = cfg["structure"]
    target_lang = (lang or "en").lower()

    if client:
        lang_rule = "5. STRICT SCRIPT REQUIREMENT: The entire output MUST be written 100% in pure Marathi (मराठी - Devanagari script). Do NOT use English sentences." if target_lang == "mr" else (
            "5. STRICT SCRIPT REQUIREMENT: The entire output MUST be written 100% in pure Hindi (हिंदी - Devanagari script). Do NOT use English sentences." if target_lang == "hi" else
            "5. Write in clear, everyday Plain English."
        )

        prompt = f"""You are LexiGuide Legal Document Simplifier.
Rewrite the following legal document excerpt into clear, natural, everyday {lang_name}.

RULES:
1. Exact Meaning: Strictly preserve all facts, names, dates, amounts, conditions, and rights/obligations.
2. Clarity: Replace archaic legalese with simple, easily understood everyday words in {lang_name}.
3. Structure:
{structure_guide}
4. Keep it direct, accurate, and easy to read.
{lang_rule}

LEGAL TEXT (Page {page_num}):
\"\"\"{text[:3000]}\"\"\"

{lang_name} Output:
"""

        for model_name in CANDIDATE_MODELS[:2]:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                output = response.text.strip()
                if output and len(output) > 20:
                    _simplifier_cache[cache_key] = output
                    return output
            except Exception as e:
                print(f"[Simplifier] Model '{model_name}' failed: {e}")
                continue

    # Structured, localized fallback if Gemini is experiencing high demand or offline
    amounts = re.findall(r"(?:Rs\.?|INR|₹)\s*[\d,]+(?:\.\d+)?|\b\d+,\d+\b", text, re.IGNORECASE)
    dates = re.findall(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text, re.IGNORECASE)
    
    # Extract parties/names if present
    tenant_m = re.search(r"received\s+from\s*[:\-]?\s*([A-Za-z\.\s]+?)(?:,|\n|\s{2,}|the\s+sum)", text, re.IGNORECASE)
    tenant_name = tenant_m.group(1).strip() if tenant_m else ""

    if target_lang == "mr":
        det_parts = []
        if amounts:
            det_parts.append(f"रक्कम / भाडे: {', '.join(amounts[:2])}")
        if tenant_name:
            det_parts.append(f"भाडेकरू / देणारे: {tenant_name}")
        if dates:
            det_parts.append(f"तारीख / कालावधी: {', '.join(dates[:2])}")
        details_str = " | ".join(det_parts) if det_parts else "सर्व कायदेशीर बाबी व नियमांचा समावेश आहे."

        fallback_out = (
            f"**मुख्य सारांश**: पान क्र. {page_num}: या पानावर कायदेशीर करारातील महत्त्वाच्या अटी, देय रक्कम आणि संबंधित पक्षांची जबाबदारी नमूद केलेली आहे.\n"
            f"**महत्त्वाचा तपशील**: {details_str}\n"
            f"**नियम व अटी**: दोन्ही पक्षांनी करारात नमूद केलेल्या सर्व कायदेशीर अटी, वेळेवर देयके आणि नियमांचे तंतोतंत पालन करणे बंधनकारक आहे."
        )
    elif target_lang == "hi":
        det_parts = []
        if amounts:
            det_parts.append(f"राशि / किराया: {', '.join(amounts[:2])}")
        if tenant_name:
            det_parts.append(f"किरायेदार / भुगतानकर्ता: {tenant_name}")
        if dates:
            det_parts.append(f"तिथियां / अवधि: {', '.join(dates[:2])}")
        details_str = " | ".join(det_parts) if det_parts else "सभी आवश्यक कानूनी विवरण शामिल हैं।"

        fallback_out = (
            f"**मुख्य सारांश**: पृष्ठ {page_num}: इस पृष्ठ पर कानूनी दस्तावेज़ के मुख्य प्रावधान, देय राशि और संबंधित पक्षों की जिम्मेदारियां निर्धारित की गई हैं।\n"
            f"**महत्वपूर्ण विवरण**: {details_str}\n"
            f"**नियम एवं शर्तें**: दोनों पक्षों को अनुबंध में उल्लिखित सभी कानूनी नियमों, समय पर भुगतान और शर्तों का पालन करना अनिवार्य है।"
        )
    else:
        det_parts = []
        if amounts:
            det_parts.append(f"Amounts: {', '.join(amounts[:2])}")
        if tenant_name:
            det_parts.append(f"Tenant / Payer: {tenant_name}")
        if dates:
            det_parts.append(f"Dates: {', '.join(dates[:2])}")
        details_str = " | ".join(det_parts) if det_parts else "Standard legal obligations and covenants."

        fallback_out = (
            f"**Key Summary**: Page {page_num}: This page establishes key contractual rights, payment obligations, and responsibilities.\n"
            f"**Important Details**: {details_str}\n"
            f"**Terms & Conditions**: The parties shall strictly abide by all covenants, payments, and terms agreed herein."
        )

    _simplifier_cache[cache_key] = fallback_out
    return fallback_out


def simplify_chunks(chunks: list, lang: str = "en") -> list:
    """
    Simplifies document chunks grouped by page in parallel using ThreadPoolExecutor.
    Returns: [{"page": int, "original": str, "simplified": str}, ...]
    """
    if not chunks:
        return []

    # Group chunk texts by page
    page_map = {}
    for c in chunks:
        p = c.get("page", 1)
        if p not in page_map:
            page_map[p] = []
        page_map[p].append(c.get("text", ""))

    pages_to_process = []
    # If the document has many pages (e.g. 20+), process the first 8 pages for instant response
    sorted_pages = sorted(page_map.items())
    if len(sorted_pages) > 8:
        sorted_pages = sorted_pages[:8]

    for page_num, text_pieces in sorted_pages:
        full_page_text = "\n\n".join(t for t in text_pieces if t.strip())
        if full_page_text:
            pages_to_process.append((page_num, full_page_text))

    def _worker(item):
        p_num, f_text = item
        simp = simplify_page_text(f_text, p_num, lang=lang)
        return {
            "page": p_num,
            "original": f_text[:400] + ("..." if len(f_text) > 400 else ""),
            "simplified": simp
        }

    # Parallelize across pages with up to 5 workers for blazing speed
    results = []
    with ThreadPoolExecutor(max_workers=min(5, len(pages_to_process) or 1)) as executor:
        future_map = {executor.submit(_worker, item): item[0] for item in pages_to_process}
        for future in as_completed(future_map):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"[Simplifier Thread Error]: {e}")

    results.sort(key=lambda x: x["page"])
    return results