"""
Q&A Module: Grounded RAG, Section Deep-Dive, and Scenario Generation.
Acts as the intelligent routing engine for user queries, ensuring high
data fidelity and strict adherence to the defined Pydantic schemas.
"""

import json
import re
from typing import Any, Optional

from openai import OpenAI

from backend import storage
from backend.config import get_settings
from backend.retrieval import CORE_SECTIONS, retrieve_for_section
from backend.schemas import (
    Citation,
    ScenarioStepOutput,
    NOT_FOUND_MESSAGE,
    ScenarioQAResponseOutput,
    QAResponseOutput,
    SectionSummaryWithConfidence
)
from backend.summarization import summarize_section
from backend.utils import load_terminology_map, normalize_text

NOT_FOUND_ANSWER = "Not found in this document."
QA_RESPONSE_DISCLAIMER = "This explanation is for informational purposes only. Refer to official policy documents."

# --- Intent Detection Patterns ---
# Regex patterns to detect when a user wants a comprehensive summary of a section
DETAIL_INTENT_PATTERNS = [
    r"more\s+detail\s+about", r"in\s+more\s+detail", r"deeper\s+summary\s+of",
    r"detailed\s+summary\s+of", r"deep\s+dive\s+(?:into|on)",
]

# Keywords that trigger the step-by-step hypothetical cost calculator
SCENARIO_TRIGGER_PHRASES = ["what would happen if", "example scenario", "how much would i pay if"]


# --- Core Q&A Helpers ---

def _qa_build_context(chunks: list[dict[str, Any]]) -> str:
    """Formats retrieved chunks into a context block with explicit citation keys."""
    parts = []
    for c in chunks:
        parts.append(
            f"---\nChunk {c.get('chunk_id', '')} (page {c.get('page_number', 0)}):\n{(c.get('chunk_text') or '').strip()}\n")
    return "\n".join(parts).strip()


def _parse_llm_json(raw: str) -> Optional[dict]:
    """Robustly extracts JSON from LLM markdown responses to prevent parsing crashes."""
    try:
        match = re.search(r"(\{.*\})", raw, re.DOTALL)
        return json.loads(match.group(1)) if match else None
    except (json.JSONDecodeError, AttributeError):
        return None


# --- Standard Factual RAG ---

def ask(doc_id: str, question: str, top_k: int = 6) -> QAResponseOutput:
    """
    Standard Grounded RAG for factual policy questions.
    Ensures every claim is cited directly from the vector store chunks.
    """
    question = (question or "").strip()
    chunks = storage.query(doc_id, question, top_k=top_k)
    allowed_ids = {str(c.get("chunk_id")) for c in chunks if c.get("chunk_id")}

    if not chunks:
        return QAResponseOutput(
            doc_id=doc_id,
            question=question,
            answer=NOT_FOUND_ANSWER,
            answer_type="not_found",
            citations=[],
            confidence="low",
            disclaimer=QA_RESPONSE_DISCLAIMER
        )

    context = _qa_build_context(chunks)
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system",
             "content": "You are a policy Q&A system. Answer using ONLY provided chunks. Cite every claim."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.1  # Low temperature ensures deterministic, factual output
    )

    parsed = _parse_llm_json(response.choices[0].message.content or "")
    if not parsed:
        return QAResponseOutput(
            doc_id=doc_id, question=question, answer=NOT_FOUND_ANSWER,
            answer_type="not_found", confidence="low", disclaimer=QA_RESPONSE_DISCLAIMER
        )

    # Clean text and Validate Citations
    term_map = load_terminology_map()
    answer_text = normalize_text(parsed.get("answer") or "", term_map)
    citations = [
        Citation(page=c['page'], chunk_id=c['chunk_id'])
        for c in parsed.get("citations", [])
        if c.get("chunk_id") in allowed_ids
    ]

    # Assign confidence based on citation presence and source chunk count
    confidence = "high" if len(citations) > 0 and len(chunks) >= 3 else "medium"
    if not citations:
        confidence = "low"

    return QAResponseOutput(
        doc_id=doc_id,
        question=question,
        answer=answer_text,
        answer_type=parsed.get("answer_type", "answerable"),
        citations=citations,
        confidence=confidence,
        disclaimer=QA_RESPONSE_DISCLAIMER
    )


# --- Scenario Generation ---

def _scenario_system_prompt() -> str:
    """Strict constraints for generating step-by-step cost scenarios."""
    return """You are an example scenario generator for a health policy. 
    Use ONLY numbers and terms from the provided chunks.
    Rules: 
    - Every dollar amount or percentage MUST have a citation. 
    - Output 3–6 steps explaining the cost flow (e.g., Deductible -> Copay/Coinsurance).
    - If info is missing, set "not_found": true.
    - Output valid JSON only."""


def ask_scenario(doc_id: str, question: str, scenario_type: str = "General") -> ScenarioQAResponseOutput:
    """
    Retrieves cost-sharing data and generates a hypothetical price breakdown.
    This logic acts as a highly specialized RAG pipeline just for numerical data.
    """
    # Specifically search for cost-sharing terms related to the scenario
    query = f"{scenario_type} deductible copay coinsurance out of pocket"
    chunks = storage.query(doc_id, query, top_k=8)
    allowed_ids = {str(c.get("chunk_id")) for c in chunks}

    if not chunks:
        return ScenarioQAResponseOutput(
            doc_id=doc_id,
            question=question,
            scenario_type=scenario_type,
            not_found_message=NOT_FOUND_MESSAGE,
            confidence="low",
            disclaimer=QA_RESPONSE_DISCLAIMER
        )

    context = _qa_build_context(chunks)
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _scenario_system_prompt()},
            {"role": "user", "content": f"Context: {context}\nScenario: {question}"}
        ],
        temperature=0.1
    )

    parsed = _parse_llm_json(response.choices[0].message.content or "")

    # If parsing fails or LLM explicitly says not found
    if not parsed or parsed.get("not_found"):
        return ScenarioQAResponseOutput(
            doc_id=doc_id,
            question=question,
            scenario_type=scenario_type,
            not_found_message=NOT_FOUND_MESSAGE,
            confidence="low",
            disclaimer=QA_RESPONSE_DISCLAIMER
        )

    # Validate steps, filter hallucinated chunk_ids, and normalize text
    term_map = load_terminology_map()
    final_steps = []

    for i, s in enumerate(parsed.get("steps", [])):
        text = normalize_text(s.get("text", ""), term_map)

        cites = []
        for c in s.get("citations", []):
            cid = c.get("chunk_id")
            if cid in allowed_ids:
                cites.append(Citation(page=c.get("page", 1), chunk_id=cid))

        final_steps.append(ScenarioStepOutput(
            step_number=s.get("step_number") or (i + 1),
            text=text,
            citations=cites
        ))

    return ScenarioQAResponseOutput(
        doc_id=doc_id,
        question=question,
        scenario_type=scenario_type,
        steps=final_steps,
        confidence="high" if len(final_steps) >= 3 else "medium",
        disclaimer=QA_RESPONSE_DISCLAIMER
    )


# --- Routing Engine ---

def _handle_section_detail(doc_id: str, question: str, section_name: str) -> QAResponseOutput:
    """Uses the summarization logic to provide a detailed section summary as a QA response."""
    chunks = retrieve_for_section(doc_id, section_name)

    # We leverage the existing summarizer which already outputs validated bullets
    summary: SectionSummaryWithConfidence = summarize_section(section_name, chunks, detail_level="detailed")

    # Flatten bullets into a single string for the QAResponseOutput 'answer' field
    answer_text = f"Detailed overview of {section_name}:\n" + "\n".join([f"- {b.text}" for b in summary.bullets])

    # Extract unique citations from bullets to pass to the UI
    all_citations = []
    seen_chunks = set()
    for b in summary.bullets:
        for c in b.citations:
            if c.chunk_id not in seen_chunks:
                all_citations.append(c)
                seen_chunks.add(c.chunk_id)

    return QAResponseOutput(
        doc_id=doc_id,
        question=question,
        answer=answer_text,
        answer_type="section_detail",
        citations=all_citations,
        confidence=summary.confidence,
        disclaimer=QA_RESPONSE_DISCLAIMER,
        validation_issues=summary.validation_issues
    )


def route_question(doc_id: str, question: str) -> Any:
    """
    Main entry point for Q&A. Routes questions based on detected intent:
    1. Scenario Intent -> ask_scenario()
    2. Section Deep-Dive -> _handle_section_detail()
    3. General QA -> ask()
    """
    q_lower = (question or "").lower()

    # 1. Detect Scenario Intent (e.g., "What if I visit the ER?")
    if any(phrase in q_lower for phrase in SCENARIO_TRIGGER_PHRASES):
        scenario_type = "ER" if "er" in q_lower or "emergency" in q_lower else "General"
        return ask_scenario(doc_id, question, scenario_type=scenario_type)

    # 2. Detect Section Deep-Dive Intent (e.g., "Tell me more about Cost Summary")
    if any(re.search(p, q_lower) for p in DETAIL_INTENT_PATTERNS):
        for section in CORE_SECTIONS:
            if section.lower() in q_lower:
                return _handle_section_detail(doc_id, question, section)

    # 3. Default to Standard Factual RAG
    return ask(doc_id, question)