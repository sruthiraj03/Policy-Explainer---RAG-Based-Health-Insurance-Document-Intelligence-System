"""
API Routes Orchestrator.
This module defines the FastAPI routers. Each router acts as a gateway
to a specific part of the PolicyExplainer pipeline.
"""

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from backend import storage
from backend.evaluation import run_all_evaluations
from backend.ingestion import run_ingest
from backend.qa import route_question  # We only need the unified router now!
from backend.retrieval import CORE_SECTIONS, retrieve_for_section
from backend.summarization import run_full_summary_pipeline, summarize_section

# --- Ingest: PDF Upload & Processing ---
router_ingest = APIRouter()


@router_ingest.post("/ingest")
async def ingest(file: UploadFile) -> dict:
    """Receives PDF, validates it, and triggers the extraction/chunking pipeline."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        content = await file.read()
        if len(content) < 4 or content[:4] != b"%PDF":
            raise HTTPException(status_code=400, detail="Invalid PDF format")

        doc_id = run_ingest(content)
        return {"doc_id": doc_id, "filename": file.filename}

    except ValueError as ve:
        # This catches your "Validation Failed" and keyword-count errors
        # We return a 400 so the frontend knows the document was rejected, not that the server broke
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        # This remains for actual system crashes (e.g., database connection issues)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


# --- Summary: LLM Extraction ---
router_summary = APIRouter()


@router_summary.post("/{doc_id}")
async def post_summary(doc_id: str) -> dict:
    """Runs the full RAG pipeline for all 6 sections defined in the schema."""
    try:
        summary = run_full_summary_pipeline(doc_id)
        return summary.model_dump()
    except Exception as e:
        import traceback             # <-- ADD THIS
        traceback.print_exc()        # <-- ADD THIS: It forces the terminal to print the red error!
        raise HTTPException(status_code=500, detail=str(e))


@router_summary.post("/{doc_id}/section/{section_id}")
async def post_section_summary(doc_id: str, section_id: str) -> dict:
    """Allows the UI to request a 'Deep Dive' (detailed level) for one specific section."""
    if section_id not in CORE_SECTIONS:
        raise HTTPException(status_code=400, detail="Invalid section ID")
    try:
        chunks = retrieve_for_section(doc_id, section_id)
        out = summarize_section(section_id, chunks, detail_level="detailed")
        return out.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Q&A: Grounded Retrieval ---
router_qa = APIRouter()


class QABody(BaseModel):
    question: str


@router_qa.post("/{doc_id}")
async def ask_endpoint(doc_id: str, body: QABody) -> dict:
    """Handles user questions using the unified Q&A router."""
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    try:
        # The unified route_question now automatically handles Scenarios, Deep-Dives, and Standard QA!
        response_obj = route_question(doc_id, question)
        return response_obj.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Evaluate: Analytics & Metrics ---
router_evaluate = APIRouter()


@router_evaluate.post("/{doc_id}")
async def evaluate(doc_id: str) -> dict:
    """Triggers the judge module to calculate faithfulness and completeness scores."""
    try:
        return run_all_evaluations(doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))