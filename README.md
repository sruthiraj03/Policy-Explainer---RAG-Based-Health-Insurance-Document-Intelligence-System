# PolicyExplainer

An AI-powered system that helps users understand complex health insurance policy documents.

## Objective

PolicyExplainer is a **document explanation tool** (not an advice chatbot). It:

- Accepts insurance policy PDFs
- Extracts and processes document text
- Generates structured summaries of key policy sections in plain English
- Supports grounded Q&A about policies with citations
- Explicitly states when information is **not** found
- Never guesses or uses external knowledge

Focus: **transparency**, **trust**, and **structured outputs**.

## Tech Stack

- **Backend:** Python, FastAPI
- **Frontend:** Streamlit
- **Vector DB:** Chroma (persistent local storage)
- **PDF:** PyMuPDF (fitz)
- **LLM:** Configurable via environment variables (OpenAI)

---

## Project Structure (Architecture)

```
PolicyExplainer/
├── backend/
│   ├── ingestion/        # PDF parsing, chunking, indexing
│   ├── retrieval/       # Vector search, section-specific queries
│   ├── summarization/   # Section summaries with citations
│   ├── qa/              # Grounded Q&A engine (normal, section deep-dive, scenario)
│   ├── evaluation/      # Metrics (faithfulness, completeness, simplicity)
│   ├── models/          # Pydantic schemas and domain models
│   ├── config.py        # Centralized env config (get_settings)
│   ├── storage/         # Vector store (Chroma), document storage (JSON/JSONL)
│   ├── utils/           # Shared utilities (normalize terms, doc cache)
│   ├── api/             # FastAPI routes (ingest, summary, qa, evaluate, chunks)
│   └── main.py          # FastAPI app entry
├── frontend/
│   └── streamlit_app/   # Streamlit UI
│       └── app.py
├── core/
│   └── sections.py      # Canonical section list
├── schema/
│   ├── terminology_map.json   # Terminology normalization
│   └── jargon_terms.json      # Jargon detection for simplicity
├── data/
│   └── documents/       # Per-document storage: {doc_id}/raw.pdf, pages.json, chunks.jsonl, Policy_summary.json, evaluation reports
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

### Architecture Overview

**Data Flow:**
1. **Ingestion:** PDF → extract pages → chunk (500–800 tokens, page-first) → save to disk → embed and index in Chroma
2. **Retrieval:** Query vector store with section-specific queries or user questions → return top-k chunks with metadata
3. **Summarization:** Retrieve chunks for section → LLM generates bullets with citations → validate citations → save summary
4. **Q&A:** Retrieve chunks for question → LLM generates answer with citations → validate → return structured response
5. **Evaluation:** Load summary and original text → compute faithfulness/completeness/simplicity → save reports

**Key Design Principles:**
- **Modular:** Each stage (ingestion, retrieval, summarization, qa) is independent
- **Grounded:** All outputs cite source chunks (page + chunk_id)
- **Validated:** Citations checked against allowed chunk_ids; bullets without citations dropped
- **Cached:** Policy summaries and chunks cached in-memory (5 min TTL) to reduce disk I/O
- **Error Handling:** Bad PDFs rejected early; empty text detected and cleaned up

---

## Setup

1. **Clone or open the project** and go to the project root:

   ```bash
   cd PolicyExplainer
   ```

2. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   ```

3. **Copy environment template and set variables:**

   ```bash
   copy .env.example .env   # Windows
   # cp .env.example .env   # macOS/Linux
   ```
   Edit `.env` and set the variables listed in **Configuration** below (at minimum: `OPENAI_API_KEY`).

4. **Run the backend** (from project root):

   ```bash
   uvicorn backend.main:app --reload
   ```
   API: http://localhost:8000 | Docs: http://localhost:8000/docs

5. **Run the Streamlit frontend** (from project root):

   ```bash
   streamlit run frontend/streamlit_app/app.py
   ```
   Frontend: http://localhost:8501

6. **Run tests:**

   ```bash
   python -m pytest
   ```

---

## Configuration

All configuration is read from the environment (e.g. `.env`). **Never hardcode secrets.**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | **Yes** | — | OpenAI API key. Must be non-empty. Not logged or exposed. |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Model used for embedding policy text chunks. |
| `LLM_MODEL` | No | `gpt-4o-mini` | Model used for summarization and Q&A. |
| `VECTOR_DB_PATH` | No | `./chroma_data` | Directory for Chroma vector DB persistence. Use empty string for in-memory only. |
| `API_BASE_URL` | No | `http://localhost:8000` | API base URL (for Streamlit frontend). |

**Using config in code:** import once and reuse:

```python
from backend.config import get_settings

settings = get_settings()
# settings.openai_api_key   # use for API calls only; never log
# settings.embedding_model
# settings.llm_model
# settings.vector_db_path
# settings.get_vector_db_path_resolved()  # Path for Chroma
```

Validation: `OPENAI_API_KEY` must be set and non-empty at startup; other fields use defaults where safe.

---

## Run Commands

### Backend (FastAPI)

```bash
# Development (auto-reload)
uvicorn backend.main:app --reload

# Production (single worker)
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# With logging
uvicorn backend.main:app --reload --log-level info
```

### Frontend (Streamlit)

```bash
# Default (http://localhost:8501)
streamlit run frontend/streamlit_app/app.py

# Custom port
streamlit run frontend/streamlit_app/app.py --server.port 8502
```

### Tests

```bash
# All tests
python -m pytest

# With coverage
python -m pytest --cov=backend

# Specific test file
python -m pytest tests/test_qa.py
```

---

## Demo Flow

### 1. Upload and Ingest PDF

**Via API:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@policy.pdf"
# Response: {"doc_id": "uuid-here", "filename": "policy.pdf"}
```

**Via Streamlit:**
- Click "Upload PDF" in sidebar
- Select PDF file
- Click "Ingest PDF"
- Note the `doc_id` (first 12 chars shown)

### 2. Generate Full Summary

**Via API:**
```bash
curl -X POST "http://localhost:8000/summary/{doc_id}"
# Returns PolicySummaryOutput JSON with all sections
```

**Via Streamlit:**
- Go to "Section Deep Dive" tab
- Select a section (e.g., "Cost Summary")
- Click "Get detailed summary"
- View bullets with citations

### 3. Ask Questions

**Via API:**
```bash
curl -X POST "http://localhost:8000/qa/{doc_id}" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is my deductible?"}'
# Returns QAResponseOutput with answer, citations, confidence
```

**Via Streamlit:**
- Go to "General Q&A" tab
- Enter question (e.g., "What is my deductible?")
- Click "Ask"
- View answer, confidence badge, citations, expandable evidence

### 4. Run Evaluation

**Via API:**
```bash
curl -X POST "http://localhost:8000/evaluate/{doc_id}"
# Returns evaluation_report.json with faithfulness/completeness/simplicity scores
```

**Via Streamlit:**
- Go to "Evaluation" tab
- Click "Run evaluation"
- View scores and any errors

---

## Limitations

### PDF Processing

- **Scanned PDFs:** Text extraction may fail if PDF contains only images (no OCR). Use OCR tools first.
- **Complex Layouts:** Tables, multi-column layouts may be extracted out of order.
- **Very Large PDFs:** Documents >100 pages may take longer to process; chunking and retrieval remain efficient.
- **Empty/Corrupt PDFs:** Rejected early with clear error messages.

### Retrieval and Summarization

- **Section Coverage:** Summaries depend on section-specific queries. If a section has no matching chunks, summary will be "not found."
- **Citation Accuracy:** Citations reference chunk_id and page; UI shows page only. Chunk boundaries may split sentences.
- **Bullet Limits:** Standard mode: 4–6 bullets; detailed: 8–12. Very long sections may be condensed.

### Q&A

- **Ambiguity:** Ambiguous questions (e.g., "What is covered?") return clarification requests, not answers.
- **Not Found:** If information is not in the document, response is "Not found in this document." (no external knowledge).
- **Conflicts:** If retrieved chunks contradict, answer_type="conflict" with explanation and citations to both pages.

### Evaluation

- **Faithfulness:** Lightweight token overlap check; may miss subtle paraphrasing.
- **Completeness:** Weighted by section importance; checklist items are heuristics.
- **Simplicity:** Flesch approximation; jargon detection uses fixed list.

---

## Reproducibility

### Deterministic Components

- **Chunking:** Deterministic (page-first, token-based splitting)
- **Retrieval:** Deterministic (Chroma vector search with fixed top_k)
- **Confidence Scoring:** Deterministic (based on citations, validation issues, retrieval strength)
- **Evaluation Metrics:** Deterministic (faithfulness, completeness, simplicity)

### Non-Deterministic Components

- **LLM Outputs:** Summarization and Q&A use LLM (temperature=0.1 for consistency, but outputs may vary)
- **Embeddings:** OpenAI embeddings are deterministic for same text, but model updates may change

### Reproducing Results

1. **Same Document:** Re-ingest same PDF → same doc_id → same chunks → same retrieval results
2. **Same Query:** Same question → same top-k chunks → similar LLM output (may vary slightly)
3. **Evaluation:** Same summary → same evaluation scores (deterministic)

**To ensure reproducibility:**
- Use same LLM model version (check `LLM_MODEL` in `.env`)
- Use same embedding model version (check `EMBEDDING_MODEL`)
- Keep same retrieval parameters (top_k, section queries)
- Document model versions in evaluation reports

### Storage Layout

Per document (`data/documents/{doc_id}/`):
- `raw.pdf` — original PDF (for reproducibility)
- `pages.json` — extracted pages (for evaluation)
- `chunks.jsonl` — chunks (for reproducibility)
- `Policy_summary.json` — generated summary (for evaluation)
- `evaluation_report.json` — evaluation scores
- `faithfulness_report.json`, `completeness_report.json`, `simplicity_report.json` — detailed metrics

---

## Evaluation Interpretation

### Faithfulness Score (0.0–1.0)

**What it measures:** How well summary bullets and Q&A sentences are supported by cited chunks.

- **Score = supported_units / total_units**
- **Unit = summary bullet OR Q&A sentence**
- **Support check:** Token overlap (≥15%) OR numeric match between bullet/sentence and chunk

**Interpretation:**
- **≥0.9:** Excellent — almost all claims are grounded
- **0.7–0.9:** Good — most claims grounded, minor gaps
- **0.5–0.7:** Fair — some unsupported claims
- **<0.5:** Poor — many unsupported claims (hallucination risk)

**Also check:**
- `hallucination_rate`: Units with no supporting chunks
- `contradiction_rate`: Units contradicting cited chunks

### Completeness Score (0.0–1.0)

**What it measures:** How well the summary covers expected policy information.

- **Weighted by section importance:**
  - Cost Summary: 35%
  - Covered Services: 30%
  - Administrative Conditions: 15%
  - Exclusions & Limitations: 10%
  - Plan Snapshot: 5%
  - Claims/Appeals: 5%
- **Checklist items per section:** Equal weight within section
- **Items without citations contribute 0**

**Interpretation:**
- **≥0.8:** Excellent — comprehensive coverage
- **0.6–0.8:** Good — most important sections covered
- **0.4–0.6:** Fair — gaps in important sections
- **<0.4:** Poor — major gaps

**Note:** "Not found" responses count as addressed (explicitly states missing info).

### Simplicity Score (0.0–1.0)

**What it measures:** How much simpler the summary is compared to original text.

- **Metrics:**
  - Average sentence length reduction
  - Jargon rate reduction
  - Flesch readability improvement
- **Score:** Normalized 0–1 (higher = simpler)

**Interpretation:**
- **≥0.7:** Excellent — much simpler than original
- **0.5–0.7:** Good — noticeably simpler
- **0.3–0.5:** Fair — some simplification
- **<0.3:** Poor — minimal simplification

**Note:** Very short original text may score lower (less room for improvement).

### Using Evaluation Reports

**Location:** `data/documents/{doc_id}/evaluation_report.json`

**Structure:**
```json
{
  "doc_id": "...",
  "faithfulness_score": 0.85,
  "completeness_score": 0.72,
  "simplicity_score": 0.68,
  "errors": []
}
```

**If errors present:** Check individual report files (`faithfulness_report.json`, etc.) for details.

**Best Practices:**
- Run evaluation after generating summary
- Compare scores across documents to identify patterns
- Low faithfulness → check citation enforcement
- Low completeness → check section queries and retrieval
- Low simplicity → check terminology normalization and LLM prompts

---

## API Endpoints

### POST /ingest
Upload PDF, ingest and index. Returns `doc_id`.

**Request:** `multipart/form-data` with `file` (PDF)
**Response:** `{"doc_id": "...", "filename": "..."}`

### GET /summary/{doc_id}
Get stored policy summary.

**Response:** `PolicySummaryOutput` JSON

### POST /summary/{doc_id}
Run full summary pipeline.

**Response:** `PolicySummaryOutput` JSON

### POST /summary/{doc_id}/section/{section_id}
Generate section-only summary (detailed mode).

**Response:** `SectionSummaryOutput` JSON

### POST /qa/{doc_id}
Answer a question.

**Request:** `{"question": "..."}`
**Response:** `QAResponseOutput` JSON (routes to section_detail or scenario if detected)

### POST /evaluate/{doc_id}
Run evaluation metrics.

**Response:** `evaluation_report.json` structure

### GET /chunks/{doc_id}
Get all chunks for evidence panel.

**Response:** `{"doc_id": "...", "chunks": [...]}`

---

## Principles

- **Modular architecture:** Ingestion → retrieval → summarization / qa
- **Clean Python typing:** Pydantic models, type hints
- **No hardcoded secrets:** All config via environment
- **No assumptions about policy content:** Works with any policy structure
- **Reproducibility:** Deterministic chunking, retrieval, evaluation
- **Reliability:** Error handling, validation, citation enforcement
- **Transparency:** Citations, confidence scores, evaluation metrics

---

## Troubleshooting

### "PDF has no extractable text"
- PDF may be scanned (images only). Use OCR first.
- PDF may be corrupted. Try opening in a PDF viewer.

### "Document not found" (404)
- Check `doc_id` is correct (UUID format)
- Verify document exists in `data/documents/{doc_id}/`

### Low faithfulness score
- Check citations in summary bullets
- Verify chunks contain cited text
- Review `faithfulness_report.json` for unsupported units

### Low completeness score
- Check section queries in `backend/retrieval/section_queries.py`
- Verify retrieval returns chunks for all sections
- Review `completeness_report.json` for missing checklist items

### API errors (500)
- Check `OPENAI_API_KEY` is set and valid
- Verify Chroma DB path is writable
- Check logs for detailed error messages

---

## License

[Add license information if applicable]
