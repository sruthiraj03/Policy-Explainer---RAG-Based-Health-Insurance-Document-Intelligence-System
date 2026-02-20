"""API routes: ingest, summary, qa, evaluate, chunks."""

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from backend import storage
from backend.evaluation import run_all
from backend.ingestion import run_ingest
from backend.qa import ask, route_question, route_scenario_question
from backend.retrieval import CORE_SECTIONS, retrieve_for_section
from backend.summarization import run_full_summary_pipeline, summarize_section

# --- Ingest ---
router_ingest = APIRouter()


@router_ingest.post("/ingest")
async def ingest(file: UploadFile) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}") from e
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) < 4 or content[:4] != b"%PDF":
        raise HTTPException(status_code=400, detail="File does not appear to be a valid PDF (missing PDF header)")
    try:
        doc_id = run_ingest(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}") from e
    return {"doc_id": doc_id, "filename": file.filename}


# --- Summary ---
router_summary = APIRouter()


@router_summary.get("/{doc_id}")
async def get_summary(doc_id: str) -> dict:
    try:
        summary = storage.load_policy_summary(doc_id)
        return summary.model_dump()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router_summary.post("/{doc_id}")
async def post_summary(doc_id: str) -> dict:
    try:
        summary = run_full_summary_pipeline(doc_id)
        return summary.model_dump()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router_summary.post("/{doc_id}/section/{section_id}")
async def post_section_summary(doc_id: str, section_id: str) -> dict:
    if section_id not in CORE_SECTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown section. Use one of: {list(CORE_SECTIONS)}")
    try:
        chunks = retrieve_for_section(doc_id, section_id)
        out = summarize_section(section_id, chunks, detail_level="detailed")
        return out.model_dump()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- Q&A ---
router_qa = APIRouter()


class QABody(BaseModel):
    question: str


@router_qa.post("/{doc_id}")
async def ask_endpoint(doc_id: str, body: QABody) -> dict:
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    try:
        section_detail = route_question(doc_id, question)
        if section_detail is not None:
            return section_detail.model_dump()
        scenario = route_scenario_question(doc_id, question)
        if scenario is not None:
            return scenario.model_dump()
        return ask(doc_id, question, top_k=6).model_dump()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- Evaluate ---
router_evaluate = APIRouter()


@router_evaluate.post("/{doc_id}")
async def evaluate(doc_id: str) -> dict:
    try:
        return run_all(doc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- Chunks ---
router_chunks = APIRouter()


@router_chunks.get("/{doc_id}")
async def get_chunks(doc_id: str) -> dict:
    try:
        chunks = storage.load_chunks(doc_id)
        return {
            "doc_id": doc_id,
            "chunks": [{"chunk_id": c.chunk_id, "page_number": c.page_number, "chunk_text": c.chunk_text} for c in chunks],
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
