# PolicyExplainer Architecture

This document describes the end-to-end architecture of PolicyExplainer, a modular Retrieval-Augmented Generation (RAG) system for insurance policy intelligence with strict grounding, citation enforcement, and evaluation-backed outputs.

PolicyExplainer is designed to prioritize reliability and transparency over open-ended generation. All outputs are restricted to the uploaded document. Unsupported claims are filtered out through deterministic validation.

---

# System Overview

PolicyExplainer is composed of two layers:

- Frontend (Streamlit): user interface, workflow orchestration, and document interactions
- Backend (FastAPI): ingestion, retrieval, LLM orchestration, citation validation, and evaluation

High-level responsibilities:

- Streamlit handles user actions (upload, summary, FAQs, Q&A, export)
- FastAPI performs all document processing and RAG operations
- Chroma stores embeddings for retrieval
- Local storage persists artifacts for reproducibility and auditing

---

# High-Level Data Flow

```mermaid
flowchart LR
  U[User] --> FE[Streamlit Frontend]
  FE -->|Upload PDF| BE[FastAPI Backend]

  subgraph Ingestion["Ingestion (Deterministic)"]
    BE --> P[PDF Parse + Clean (PyMuPDF)]
    P --> C[Chunking + Overlap]
    C --> FS[(Local Artifact Store)]
    C --> E[Embeddings]
    E --> VDB[(Chroma Vector DB)]
  end

  subgraph Retrieval["Retrieval (Deterministic)"]
    VDB --> R[Section-Aware Multi-Query Retrieval]
    R --> D[Deduplicate + Sort by Document Order]
  end

  subgraph Generation["Generation (Probabilistic)"]
    D --> LLM[LLM Structured JSON Output]
  end

  subgraph Validation["Validation + Scoring (Deterministic)"]
    LLM --> CV[Citation Validator]
    CV --> CS[Confidence Scoring]
    CV --> EV[Evaluation Metrics]
  end

  CS --> FE
  EV --> FE
  FE -->|Render Results| U
```

---

# Component Breakdown

## Frontend (Streamlit)

Primary responsibilities:

- Policy upload experience
- Navigation: Summary, New Policy, Save Insurance Summary, FAQs
- Policy Assistant chat interface
- Evidence display (citations and page references)
- Exporting summaries to PDF

Frontend modules typically include:

- UI components (hero, dashboard, sidebar, chat)
- State management (current doc_id, selected section, chat history)
- Styling and layout utilities
- PDF export utilities (summary-to-PDF formatting)

Design goal:
- Keep frontend focused on UX and orchestration
- Keep all retrieval and generation logic in the backend for auditability and consistency

---

## Backend (FastAPI)

The backend owns the system’s core logic. It is responsible for:

- Ingestion: parse and clean PDF text, chunk deterministically, persist artifacts
- Retrieval: vector search with section-aware queries, deduplicate and order chunks
- Summarization: structured section summaries with citations
- Q&A: grounded answers with citation enforcement and confidence scoring
- FAQs: policy-specific question generation grounded in the document
- Evaluation: scoring outputs on faithfulness, completeness, and simplicity

Key backend modules:

- ingestion.py: parsing, cleaning, chunking, indexing
- retrieval.py: section-aware retrieval logic
- summarization.py: section summaries and structured generation contracts
- qa.py: grounded Q&A pipeline
- evaluation.py: deterministic scoring and reports
- schemas.py: Pydantic schemas and structured output contracts
- storage.py: persistence layer (local artifacts + vector DB coordination)
- utils.py: shared utilities and normalization helpers
- api.py: route definitions
- main.py: application entrypoint and wiring

---

# Ingestion Pipeline

The ingestion stage is designed to be deterministic and reproducible.

1. PDF parsing
   - Extract text per page using PyMuPDF
   - Clean and normalize extracted content
   - Detect empty or unusable documents early

2. Chunking
   - Token-based chunking (approximately 500–800 tokens)
   - Sliding overlap (~80 tokens) to preserve context
   - Chunk IDs are deterministic:

   ```text
   c_{page_number}_{chunk_index}
   ```

3. Persistence
   - Store all intermediate artifacts to local storage
   - Embed chunks and index them into Chroma for retrieval

Outputs produced by ingestion:

- pages.json: extracted text per page
- chunks.jsonl: chunk objects with IDs and metadata
- raw.pdf: original file stored for reproducibility
- vector index entries: embeddings stored in Chroma

---

# Storage Strategy

PolicyExplainer uses a dual storage strategy for traceability and reproducibility:

1. Local artifact store (source-of-truth)
   - Used for audit, debugging, evaluation, and reproducibility
   - Stores raw and processed content

2. Vector database (Chroma)
   - Used for fast semantic retrieval
   - Stores embeddings and metadata fields (doc_id, page_number, chunk_id)

Local artifact layout:

```text
data/documents/{doc_id}/
├─ raw.pdf
├─ pages.json
├─ chunks.jsonl
├─ policy_summary.json
└─ evaluation_report.json
```

Vector store:

```text
./chroma_data
```

---

# Retrieval Layer

Retrieval is designed to maximize recall while keeping context bounded and stable for LLM generation.

Instead of issuing a single query, the system uses section-aware multi-query retrieval.

Example: Cost Summary might issue separate retrieval queries for:

- deductible
- copay
- coinsurance
- out-of-pocket maximum
- premium

Retrieval algorithm:

1. Run vector search for each sub-query
2. Merge retrieved results
3. Deduplicate by chunk_id (retain best match)
4. Sort by document order (page_number, chunk_index)
5. Cap context length to avoid prompt overflow and reduce noise

Benefits:

- Higher recall for sparse policy terminology
- Reduced risk of missing key subsections
- Improves LLM consistency by presenting context in document order
- Mitigates the “Lost-in-the-Middle” issue

---

# Generation Layer

Generation is the only probabilistic component of the pipeline.

The model is required to output structured JSON (not free-form prose).

All generation tasks follow the same pattern:

- Construct section- or task-specific prompt
- Provide retrieved chunks as bounded context
- Enforce JSON schema output
- Return citations as chunk_id references

Generation use cases:

- Section summaries
- Policy FAQs
- Grounded Q&A answers

---

# Citation Validation Layer

Citation validation is deterministic and enforced after generation.

Goals:

- Prevent unsupported claims from appearing in output
- Enforce traceability back to the document
- Provide confidence scoring signals

Validation logic:

- Only allow citations that reference retrieved chunk_ids
- Drop bullets or answer sentences that have no valid citations
- Normalize page references and chunk IDs
- Record validation warnings for evaluation reporting

If retrieval returns no relevant chunks, the system must return:

```text
Not found in this document.
```

The system does not guess or rely on external knowledge.

---

# Confidence Scoring

Confidence is derived from deterministic signals such as:

- Citation density (how well-supported the output is)
- Citation validity (chunk IDs exist and match retrieved set)
- Retrieval strength (similarity distributions and coverage)
- Validation issues (dropped bullets, invalid citations, missing support)

Confidence is intended to reflect reliability relative to the uploaded document, not legal correctness.

---

# Evaluation Framework

Evaluation is deterministic and run post-generation.

The system measures output quality across three dimensions:

## 1. Faithfulness (0.0 – 1.0)

Measures whether generated bullets or answer units are supported by their cited chunks.

Support checks may include:

- Token overlap thresholds
- Numeric consistency checks (deductibles, limits, copays)

Higher score indicates stronger grounding.

---

## 2. Completeness (0.0 – 1.0)

Measures coverage across canonical policy sections.

Weighted scoring encourages focus on high-impact sections.

Example weights:

- Cost Summary (35%)
- Covered Services (30%)
- Administrative Conditions (15%)
- Exclusions & Limitations (10%)
- Plan Snapshot (5%)
- Claims & Appeals (5%)

Higher score indicates broader coverage.

---

## 3. Simplicity (0.0 – 1.0)

Measures how much easier the summary is to understand compared to the source document.

Simplicity is designed to capture clarity improvements, not just brevity.

Recommended structure:

- Readability improvement (Flesch delta, sentence length reduction)
- Jargon reduction (domain term frequency reduction using a fixed jargon list)
- Structural clarity (bullets vs dense paragraphs, clause complexity)

Example scoring composition:

```text
Simplicity Score =
0.4 * readability_improvement
+ 0.4 * jargon_reduction
+ 0.2 * structural_clarity
```

Higher score indicates a clearer explanation for non-technical users.

---

# Reproducibility Notes

Deterministic components:

- Chunking
- Retrieval ordering and deduplication
- Citation validation logic
- Confidence scoring signals
- Faithfulness, completeness, and simplicity scoring

Non-deterministic components:

- LLM outputs (summaries and answers), though temperature is typically kept low for stability

Reproducibility best practices:

- Persist raw.pdf, pages.json, chunks.jsonl
- Fix model versions in configuration
- Keep retrieval parameters stable (top_k, query sets)
- Store evaluation reports per document

---

# Security and Deployment Considerations

Recommended production hardening:

- Restrict CORS origins
- Add authentication / rate limiting for API endpoints
- Avoid logging sensitive user text or policy content
- Store API keys only in environment variables
- Implement retention policies for uploaded documents

---

# Summary

PolicyExplainer is designed as an end-to-end RAG system with:

- Deterministic ingestion and retrieval
- Structured generation with strict output schemas
- Citation enforcement and confidence scoring
- Evaluation metrics for faithfulness, completeness, and simplicity
- Artifact persistence for auditability and reproducibility

This architecture prioritizes grounded, transparent, and measurable outputs over open-ended generation.

---

End of Architecture Document.
