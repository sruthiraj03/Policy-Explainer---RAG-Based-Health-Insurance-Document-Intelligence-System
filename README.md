# PolicyExplainer

PolicyExplainer is an AI-powered document intelligence system that helps users understand complex health insurance policy PDFs through structured summaries, grounded Q&A, and automated evaluation metrics.

This system is a document explanation tool — not medical or legal advice.  
It answers strictly using the uploaded document and explicitly states when information is not found.  
It never uses external knowledge.

---

## Overview

Insurance policies are long, dense, and difficult to interpret. PolicyExplainer transforms unstructured policy PDFs into structured, explainable outputs using a modular Retrieval-Augmented Generation (RAG) architecture.

The system performs:

- PDF ingestion and cleaning  
- Token-based chunking with overlap  
- Embedding generation and vector indexing  
- Multi-query section retrieval  
- Citation-enforced summarization  
- Grounded Q&A with routing logic  
- Automated evaluation metrics (faithfulness, completeness)  
- Confidence scoring  

---

## Core Features

### 1. PDF Ingestion Pipeline

- Upload insurance policy PDF  
- Extract text per page using PyMuPDF  
- Remove repeated headers and footers  
- Validate likely policy structure using keyword heuristics  
- Chunk text into 500–800 token windows with overlap  
- Persist:
  - raw.pdf  
  - pages.json  
  - chunks.jsonl  
- Generate embeddings and index in Chroma vector database  

Chunk IDs follow the format:

```text
c_{page_number}_{chunk_index}
```

---

### 2. Structured Policy Summaries

Generates plain-English bullet summaries across six canonical sections:

- Plan Snapshot  
- Cost Summary  
- Summary of Covered Services  
- Administrative Conditions  
- Exclusions & Limitations  
- Claims, Appeals & Member Rights  

Each section:

- Returns a `present` flag  
- Includes structured bullet objects  
- Enforces citation validation  
- Drops bullets without valid citations  
- Computes a confidence score (High / Medium / Low)  

All citations are filtered to ensure they reference retrieved chunk_ids only.

---

### 3. Grounded Q&A (RAG)

Users can ask natural language questions such as:

- What is my deductible?  
- Is urgent care covered?  
- Do I need prior authorization?  

The Q&A pipeline:

1. Retrieve top-k chunks using vector search  
2. Sort chunks in document order  
3. Force structured JSON response  
4. Validate citations against allowed chunk_ids  
5. Remove unsupported claims  
6. Compute confidence score  

If information is not present in the document, the system returns exactly:

```text
Not found in this document.
```

No external knowledge is ever used.

---

### 4. Question Routing Logic

The system routes user input into one of four paths:

- Greeting response (no citations)  
- Scenario question (e.g., emergency visit)  
- Section deep-dive explanation  
- Standard RAG question  

This ensures appropriate response style and grounding behavior.

---

### 5. Evaluation Metrics

The system supports automated evaluation via:

```text
POST /evaluate/{doc_id}
```

Metrics include:

- Faithfulness (citation support validation)  
- Completeness (section coverage weighting)  
- Structural validation checks  

Faithfulness ensures summary bullets are supported by cited chunks.  
Completeness ensures major policy sections are addressed.  
Confidence scores reflect citation density and validation issues.

---

## Architecture

High-Level Flow:

1. Upload PDF  
2. Extract and clean page text  
3. Chunk text with overlap  
4. Store chunks locally  
5. Generate embeddings  
6. Index in Chroma  
7. Retrieve per section  
8. Generate LLM summary or Q&A  
9. Validate citations  
10. Return structured output  
11. Optionally evaluate  

Backend modules:

- ingestion.py  
- retrieval.py  
- summarization.py  
- qa.py  
- evaluation.py  
- storage.py  
- schemas.py  
- utils.py  

The frontend is built in Streamlit and communicates with the FastAPI backend.

---

## Tech Stack

Backend:
- Python  
- FastAPI  
- Pydantic  
- PyMuPDF  
- Chroma vector DB  
- OpenAI API  

Frontend:
- Streamlit  
- Modular component structure  

Persistence:
- JSON storage per document  
- Chroma persistent vector index  

---

## Project Structure

```text
PolicyExplainer/
├─ backend/
│  ├─ api.py
│  ├─ config.py
│  ├─ ingestion.py
│  ├─ retrieval.py
│  ├─ summarization.py
│  ├─ qa.py
│  ├─ evaluation.py
│  ├─ schemas.py
│  ├─ storage.py
│  └─ utils.py
├─ frontend/
│  ├─ app.py
│  ├─ assets/
│  │  └─ header_image.jpg
│  ├─ components/
│  │  ├─ chat.py
│  │  ├─ dashboard.py
│  │  ├─ hero.py
│  │  └─ sidebar.py
│  └─ utils/
│     ├─ pdf_generator.py
│     ├─ state.py
│     └─ style.py
├─ schema/
│  ├─ jargon_terms.json
│  ├─ summary_schema.json
│  └─ terminology_map.json
├─ tests/
├─ .env.example
├─ .gitignore
├─ pyproject.toml
├─ requirements.txt
└─ README.md
```

---

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 2. Configure Environment

Copy example:

```bash
copy .env.example .env
```

Required:

```text
OPENAI_API_KEY=your_api_key_here
```

Optional:

```text
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DB_PATH=./chroma_data
```

---

### 3. Run Backend

```bash
uvicorn backend.main:app --reload
```

Backend:
http://127.0.0.1:8000

Docs:
http://127.0.0.1:8000/docs

---

### 4. Run Frontend

```bash
streamlit run frontend/app.py
```

Frontend:
http://localhost:8501

---

## API Endpoints

```text
POST   /ingest
POST   /summary/{doc_id}
POST   /summary/{doc_id}/section/{section_id}
POST   /qa/{doc_id}
POST   /evaluate/{doc_id}
GET    /chunks/{doc_id}
```

---

## Design Principles

- Grounded outputs only  
- Strict citation enforcement  
- Deterministic chunking  
- Modular architecture  
- Environment-based configuration  
- No hardcoded secrets  
- Reproducibility via stored raw PDF + chunks  

---

## Limitations

- Scanned PDFs require OCR  
- Complex table layouts may extract imperfectly  
- Policy validation is heuristic-based  
- CORS currently open for local development  
- LLM outputs may vary slightly due to temperature  

---

## Future Improvements

- Add document deletion endpoint  
- Add stricter CORS controls  
- Add debug logging flag  
- Improve table extraction  
- Add OCR fallback support  
- Add production deployment config  

---

## License

MIT License
