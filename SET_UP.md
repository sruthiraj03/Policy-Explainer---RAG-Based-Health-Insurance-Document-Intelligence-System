# Setup Guide

This document explains how to install, configure, and run PolicyExplainer locally.

PolicyExplainer consists of:

- FastAPI backend (document ingestion, retrieval, summarization, Q&A, evaluation)
- Streamlit frontend (UI and interaction layer)
- ChromaDB vector database (embedding storage and retrieval)
- Local artifact storage per document (for reproducibility)

---

# System Requirements

- Python 3.10 or 3.11 (recommended for compatibility)
- pip
- Virtual environment support
- OpenAI API key

Optional (recommended):

- Git
- VS Code or similar IDE

---

# 1. Clone the Repository

```bash
git clone <your-repo-url>
cd PolicyExplainer
```

---

# 2. Create Virtual Environment

Create a virtual environment:

```bash
python -m venv .venv
```

Activate:

**Windows**

```bash
.venv\Scripts\activate
```

**Mac/Linux**

```bash
source .venv/bin/activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Optional (if using readability-based simplicity scoring):

```bash
pip install textstat
```

---

# 4. Configure Environment Variables

Copy the template:

**Windows**

```bash
copy .env.example .env
```

**Mac/Linux**

```bash
cp .env.example .env
```

Edit `.env`:

```text
OPENAI_API_KEY=your_api_key_here
```

Optional configuration:

```text
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DB_PATH=./chroma_data
API_BASE_URL=http://127.0.0.1:8000
```

Notes:

* `OPENAI_API_KEY` is required
* Do not hardcode secrets in code
* `.env` should never be committed

---

# 5. Run Backend (FastAPI)

From project root:

```bash
uvicorn backend.main:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

# 6. Run Frontend (Streamlit)

In a new terminal (activate virtual environment):

```bash
streamlit run frontend/app.py
```

Frontend runs at:

```text
http://localhost:8501
```

---

# 7. End-to-End Verification

1. Upload a health insurance policy PDF
2. Confirm a `doc_id` is generated
3. Generate a structured summary
4. Ask a question using Policy Assistant
5. Run evaluation

Evaluation endpoint:

```text
POST /evaluate/{doc_id}
```

Expected outputs:

* Faithfulness score
* Completeness score
* Simplicity score
* Validation results
* Confidence signals

---

# Project Storage Layout

After ingestion, artifacts are stored per document:

```text
data/documents/{doc_id}/
├─ raw.pdf
├─ pages.json
├─ chunks.jsonl
├─ policy_summary.json
└─ evaluation_report.json
```

Vector storage:

```text
./chroma_data
```

These enable reproducibility and traceability.

---

# Backend Execution Flow (What Happens Internally)

**When you upload a document:**

1. PDF is parsed and cleaned (PyMuPDF)
2. Text is split into deterministic chunks
3. Chunk artifacts are saved locally
4. Embeddings are generated and stored in ChromaDB

**When generating a summary:**

1. Section-aware multi-query retrieval runs
2. Relevant chunks are deduplicated and ordered
3. LLM generates structured JSON output
4. Citation validation filters unsupported content
5. Confidence score is computed

**When running evaluation:**

1. Summary is re-checked against source chunks
2. Faithfulness, completeness, simplicity are computed
3. Results are stored as `evaluation_report.json`

---

# Running Evaluation via CLI

```bash
curl -X POST http://127.0.0.1:8000/evaluate/{doc_id}
```

Evaluation includes:

* Faithfulness scoring
* Completeness scoring
* Simplicity scoring
* Citation validation checks

---

# Optional: Enable Simplicity Metric

Install dependency:

```bash
pip install textstat
```

Simplicity may include:

* Flesch Reading Ease improvement
* Sentence length reduction
* Jargon reduction
* Structural simplification

Keep evaluation deterministic.

---

# Common Issues

## 1. OPENAI_API_KEY not set

Error:

```text
Missing API key
```

Fix:

* Ensure `.env` exists
* Add valid API key
* Restart backend

## 2. PDF has no extractable text

Cause:

* Scanned image-based PDF

Fix:

* Run OCR before uploading

## 3. ChromaDB errors or missing index

Fix:

* Delete `./chroma_data` and re-run ingestion
* Ensure embedding model is consistent

## 4. Port already in use

Change backend port:

```bash
uvicorn backend.main:app --reload --port 8001
```

Change frontend port:

```bash
streamlit run frontend/app.py --server.port 8502
```

## 5. Inconsistent results across runs

Possible causes:

* Different model versions
* Changed chunking logic
* Cleared vector database

Fix:

* Keep environment variables consistent
* Avoid modifying ingestion logic mid-run

---

# Production Considerations

For deployment:

* Restrict CORS origins
* Add authentication to APIs
* Remove debug logs
* Use fixed model versions
* Containerize with Docker
* Add CI/CD pipeline
* Monitor API usage and latency

---

# Summary

PolicyExplainer runs locally using:

* FastAPI backend for document intelligence
* Streamlit frontend for interaction
* ChromaDB for semantic retrieval
* Local artifact storage for reproducibility

The system is modular, deterministic where required, and designed for reliable, grounded outputs.

---

*End of Setup Guide.*
