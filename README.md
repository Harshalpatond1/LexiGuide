# LexiGuide ⚖️
**AI-Powered Legal Document Assistant & Simplifier**

LexiGuide is an NLP and AI-driven assistant designed to help users understand complex legal documents, ask questions based on uploaded legal PDFs, and get simplified plain-language explanations.

---

## 🚀 Features

- 📄 **PDF Processing & Chunking**: Extracts and segments legal documents into semantic sections.
- 🔍 **Legal QA & Retrieval**: Search and query through legal documents using semantic search and retrieval algorithms.
- 💡 **AI Simplifier**: Explains complex legal terminology and clauses into easy-to-understand language powered by Google Gemini.
- 🌐 **Web Interface**: Clean, modern web UI for document uploading and interactive chat.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **NLP / AI**: Google Generative AI (Gemini), Sentence Transformers / Custom Retrieval
- **Frontend**: HTML5, CSS3, JavaScript

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Lexiguide.git
   cd Lexiguide
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**:
   Copy `.env.example` to `.env` and add your Google Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key
   ```

5. **Run the Application**:
   ```bash
   python server.py
   ```
   Open your browser and navigate to `http://localhost:5000`.
