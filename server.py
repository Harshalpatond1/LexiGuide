import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
from pdf_processor import extract_text_from_pdf
from chunker import split_into_chunks
from retriever import create_index, search
from qa_engine import extract_answer, _get_gemini, GEMINI_MODELS
from simplifier import simplify_chunks

app = Flask(__name__, static_folder="static")
app.secret_key = "lexiguide-secret-2024-secure-key"

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory document store with fallback for resilience
_store = {}


def _get_active_doc():
    sid = session.get("sid")
    if sid and sid in _store:
        return _store[sid]
    return _store.get("latest")


def _detect_doc_intelligence(text: str) -> dict:
    """Analyze document type and suggestions with instant heuristic detection and fast Gemini enrichment."""
    t_lower = text[:3000].lower()
    
    # 1. Instant heuristic detection
    if "rent receipt" in t_lower:
        doc_type = "Rent Receipt"
        suggestions = ["What is the rent amount?", "Who is the tenant?", "Who is the landlord?", "What is the receipt date?", "Are there any arrears?"]
    elif "leave and license" in t_lower or "rent agreement" in t_lower or "tenancy" in t_lower:
        doc_type = "Rent / Lease Agreement"
        suggestions = ["Who is the landlord and tenant?", "What is the monthly rent and deposit?", "What is the lease period?", "What are the termination conditions?", "Who pays maintenance and utility bills?"]
    elif "sale deed" in t_lower:
        doc_type = "Sale Deed"
        suggestions = ["Who is the buyer and seller?", "What is the sale consideration amount?", "What is the property schedule and address?", "Is the property free from encumbrances?", "When was possession handed over?"]
    elif "power of attorney" in t_lower:
        doc_type = "Power of Attorney"
        suggestions = ["Who is the principal?", "Who is appointed as attorney?", "What powers are granted?", "Can the attorney sell or mortgage property?", "Is this power of attorney revocable?"]
    elif "affidavit" in t_lower:
        doc_type = "Affidavit / Declaration"
        suggestions = ["Who is the deponent?", "What is being declared or affirmed?", "Before whom was it sworn?", "What is the date of verification?", "What is the purpose of this affidavit?"]
    elif "will" in t_lower and ("testator" in t_lower or "bequeath" in t_lower):
        doc_type = "Will & Testament"
        suggestions = ["Who is the testator?", "Who are the named beneficiaries?", "What properties/assets are bequeathed?", "Who is appointed as executor?", "Who are the attesting witnesses?"]
    elif "non-disclosure" in t_lower or "nda" in t_lower or "confidential" in t_lower:
        doc_type = "Non-Disclosure Agreement (NDA)"
        suggestions = ["Who are the disclosing and receiving parties?", "What is defined as Confidential Information?", "What is the duration of confidentiality?", "What are the permitted disclosures?", "What is the governing law?"]
    elif "employment" in t_lower or "appointment" in t_lower:
        doc_type = "Employment Agreement"
        suggestions = ["What is the employee's designation/role?", "What is the compensation/salary structure?", "What is the notice period for resignation?", "What are the confidentiality obligations?", "What is the probation period?"]
    elif "partnership" in t_lower:
        doc_type = "Partnership Deed"
        suggestions = ["Who are the partners?", "What is the profit/loss sharing ratio?", "What is each partner's capital contribution?", "How can the partnership be dissolved?", "Who manages banking and operations?"]
    else:
        doc_type = "Legal Document"
        suggestions = [
            "What are the key terms?",
            "Who are the parties involved?",
            "What is the effective date?",
            "What are the payment or financial obligations?",
            "What are the termination conditions?"
        ]

    first_line = [l.strip() for l in text.split("\n") if l.strip()][:3]
    summary = f"This document appears to be a {doc_type} ({' - '.join(first_line)[:150]}). Analyzed for Q&A and Plain-Language Simplification."

    default_meta = {
        "doc_type": doc_type,
        "summary": summary,
        "suggestions": suggestions
    }

    client = _get_gemini()
    if not client:
        return default_meta

    # Try fast enrichment from primary model only (no multi-model stall)
    prompt = f"""Analyze this legal document text excerpt and provide metadata in JSON.
DOCUMENT TEXT:
\"\"\"{text[:2000]}\"\"\"

Generate JSON:
{{
  "doc_type": "{doc_type}",
  "summary": "2-sentence executive summary of the document",
  "suggestions": {json.dumps(suggestions)}
}}
"""
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODELS[0],
            contents=prompt,
            config=dict(response_mime_type="application/json")
        )
        raw = resp.text.strip()
        data = json.loads(raw)
        if isinstance(data, dict) and "doc_type" in data:
            return data
    except Exception as e:
        print(f"[DocIntel Fast Fallback]: {e}")

    return default_meta


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = secure_filename(f.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(path)

    try:
        pages = extract_text_from_pdf(path)
        if not pages:
            return jsonify({"error": "No readable text found in this PDF"}), 400

        chunks = split_into_chunks(pages)
        index = create_index(chunks)

        # Quick document intelligence analysis
        combined_text = "\n".join(p["text"] for p in pages[:2])
        meta = _detect_doc_intelligence(combined_text)

    except Exception as e:
        print(f"[Upload Error]: {e}")
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500

    sid = session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["sid"] = sid

    doc_data = {
        "filename": filename,
        "pages": pages,
        "chunks": chunks,
        "index": index,
        "meta": meta
    }

    _store[sid] = doc_data
    _store["latest"] = doc_data

    return jsonify({
        "message": "ok",
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "doc_type": meta.get("doc_type", "Legal Document"),
        "summary": meta.get("summary", ""),
        "suggestions": meta.get("suggestions", [])
    })


@app.route("/api/sample/<filename>", methods=["POST"])
def load_sample(filename):
    allowed = ["rent-receipt_1.pdf", "rent_receipt_2test.pdf"]
    if filename not in allowed:
        return jsonify({"error": "Sample document not found"}), 404

    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": f"File {filename} not found on server"}), 404

    try:
        pages = extract_text_from_pdf(path)
        chunks = split_into_chunks(pages)
        index = create_index(chunks)
        combined_text = "\n".join(p["text"] for p in pages[:2])
        meta = _detect_doc_intelligence(combined_text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    sid = session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["sid"] = sid

    doc_data = {
        "filename": filename,
        "pages": pages,
        "chunks": chunks,
        "index": index,
        "meta": meta
    }

    _store[sid] = doc_data
    _store["latest"] = doc_data

    return jsonify({
        "message": "ok",
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "doc_type": meta.get("doc_type", "Legal Document"),
        "summary": meta.get("summary", ""),
        "suggestions": meta.get("suggestions", [])
    })


@app.route("/api/document", methods=["GET"])
def get_current_document():
    doc = _get_active_doc()
    if not doc:
        return jsonify({"active": False})

    return jsonify({
        "active": True,
        "filename": doc["filename"],
        "pages": len(doc["pages"]),
        "chunks": len(doc["chunks"]),
        "doc_type": doc.get("meta", {}).get("doc_type", "Legal Document"),
        "summary": doc.get("meta", {}).get("summary", ""),
        "suggestions": doc.get("meta", {}).get("suggestions", [])
    })


_ask_cache = {}
_simplify_cache_store = {}


@app.route("/api/ask", methods=["POST"])
def ask():
    doc = _get_active_doc()
    if not doc:
        return jsonify({"error": "No document uploaded yet. Please upload a PDF first."}), 400

    body = request.get_json(force=True, silent=True) or {}
    question = (body.get("question") or "").strip()
    ui_lang = (body.get("lang") or "en").strip().lower()
    if ui_lang not in ["en", "hi", "mr"]:
        ui_lang = "en"

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Auto-detect language of question, respecting UI language choice as fallback
    from qa_engine import detect_question_language
    effective_lang = detect_question_language(question, fallback_lang=ui_lang)

    cache_key = (doc.get("filename", ""), question.lower(), effective_lang)
    if cache_key in _ask_cache:
        return jsonify(_ask_cache[cache_key])

    try:
        results = search(question, doc["chunks"], doc["index"], top_k=5)
        result = extract_answer(question, results, pages=doc.get("pages"), lang=effective_lang)
    except Exception as e:
        print(f"[Ask Error]: {e}")
        return jsonify({"error": f"Error analyzing question: {str(e)}"}), 500

    ans_lang = result.get("lang", effective_lang)
    resp_payload = {
        "answer": result["answer"],
        "page": result["page"],
        "clause": result["clause"],
        "confidence": result["confidence"],
        "lang": ans_lang
    }
    _ask_cache[cache_key] = resp_payload
    return jsonify(resp_payload)


@app.route("/api/simplify", methods=["POST"])
def simplify():
    doc = _get_active_doc()
    if not doc:
        return jsonify({"error": "No document uploaded yet. Please upload a PDF first."}), 400

    body = request.get_json(force=True, silent=True) or {}
    lang = (body.get("lang") or "en").strip().lower()
    if lang not in ["en", "hi", "mr"]:
        lang = "en"

    cache_key = (doc.get("filename", ""), lang)
    if cache_key in _simplify_cache_store:
        return jsonify(_simplify_cache_store[cache_key])

    try:
        simplified = simplify_chunks(doc["chunks"], lang=lang)
    except Exception as e:
        print(f"[Simplify Error]: {e}")
        return jsonify({"error": f"Simplification failed: {str(e)}"}), 500

    output = [
        {
            "page": c.get("page", 1),
            "original": c.get("original", ""),
            "text": c.get("simplified", "") or c.get("original", "")
        }
        for c in simplified
    ]
    resp_payload = {"chunks": output, "lang": lang}
    _simplify_cache_store[cache_key] = resp_payload
    return jsonify(resp_payload)


@app.route("/api/clauses", methods=["GET"])
def get_clauses():
    doc = _get_active_doc()
    if not doc:
        return jsonify({"error": "No document uploaded yet"}), 400

    clauses_data = [
        {"page": p["page"], "text": p["text"]}
        for p in doc.get("pages", [])
    ]
    return jsonify({"pages": clauses_data})


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
