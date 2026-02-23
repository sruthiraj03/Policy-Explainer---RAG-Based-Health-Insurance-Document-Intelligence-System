"""
Q&A Module: Grounded RAG, Section Deep-Dive, and Scenario Generation.
Acts as the intelligent routing engine for user queries, ensuring high
data fidelity and strict adherence to the defined Pydantic schemas.
"""

"""
Q&A Module: Grounded RAG, Section Deep-Dive, and Scenario Generation.
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

DETAIL_INTENT_PATTERNS = [
    r"more\s+detail\s+about", r"in\s+more\s+detail", r"deeper\s+summary\s+of",
    r"detailed\s+summary\s+of", r"deep\s+dive\s+(?:into|on)",
]

SCENARIO_TRIGGER_PHRASES = ["what would happen if", "example scenario", "how much would i pay if"]


def _qa_build_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"---\nChunk {c.get('chunk_id', '')} (page {c.get('page_number', 0)}):\n{(c.get('chunk_text') or '').strip()}\n")
    return "\n".join(parts).strip()


def _parse_llm_json(raw: str) -> dict:
    """Aggressively strips markdown to ensure JSON parses."""
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    try:
        return json.loads(raw.strip())
    except Exception:
        match = re.search(r"(\{.*\})", raw, re.DOTALL)
        return json.loads(match.group(1)) if match else {}


def ask(doc_id: str, question: str, top_k: int = 6) -> QAResponseOutput:
    question = (question or "").strip()
    chunks = storage.query(doc_id, question, top_k=top_k)
    allowed_ids = {str(c.get("chunk_id")) for c in chunks if c.get("chunk_id")}

    if not chunks:
        return QAResponseOutput(
            doc_id=doc_id, question=question, answer=NOT_FOUND_ANSWER,
            answer_type="not_found", citations=[], confidence="low", disclaimer=QA_RESPONSE_DISCLAIMER
        )

    context = _qa_build_context(chunks)
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=settings.llm_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": 'You are a Q&A system. Answer using ONLY provided chunks. Output ONLY valid JSON containing: {"answer": "your text", "answer_type": "answerable", "citations": [{"chunk_id": "c_1_0", "page": 1}]}'},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.1
    )

    parsed = _parse_llm_json(response.choices[0].message.content or "")

    term_map = load_terminology_map()
    answer_text = normalize_text(parsed.get("answer") or NOT_FOUND_ANSWER, term_map)
    citations = [
        Citation(page=c.get('page', 0), chunk_id=c.get('chunk_id', ''))
        for c in parsed.get("citations", [])
        if c.get("chunk_id") in allowed_ids
    ]

    confidence = "high" if citations and len(chunks) >= 3 else "medium" if citations else "low"

    return QAResponseOutput(
        doc_id=doc_id, question=question, answer=answer_text,
        answer_type=parsed.get("answer_type", "answerable") if citations else "not_found",
        citations=citations, confidence=confidence, disclaimer=QA_RESPONSE_DISCLAIMER
    )


def ask_scenario(doc_id: str, question: str, scenario_type: str = "General") -> ScenarioQAResponseOutput:
    query = f"{scenario_type} deductible copay coinsurance out of pocket"
    chunks = storage.query(doc_id, query, top_k=8)
    allowed_ids = {str(c.get("chunk_id")) for c in chunks}

    if not chunks:
        return ScenarioQAResponseOutput(
            doc_id=doc_id, question=question, scenario_type=scenario_type,
            not_found_message=NOT_FOUND_MESSAGE, confidence="low", disclaimer=QA_RESPONSE_DISCLAIMER
        )

    context = _qa_build_context(chunks)
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=settings.llm_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": 'You generate hypothetical cost scenarios. Use ONLY provided chunks. Output ONLY valid JSON containing: {"steps": [{"step_number": 1, "text": "description", "citations": [{"chunk_id": "c_1_0", "page": 1}]}], "not_found": false}'},
            {"role": "user", "content": f"Context: {context}\nScenario: {question}"}
        ],
        temperature=0.1
    )

    parsed = _parse_llm_json(response.choices[0].message.content or "")

    if not parsed or parsed.get("not_found"):
        return ScenarioQAResponseOutput(
            doc_id=doc_id, question=question, scenario_type=scenario_type,
            not_found_message=NOT_FOUND_MESSAGE, confidence="low", disclaimer=QA_RESPONSE_DISCLAIMER
        )

    term_map = load_terminology_map()
    final_steps = []

    for i, s in enumerate(parsed.get("steps", [])):
        text = normalize_text(s.get("text", ""), term_map)
        cites = [Citation(page=c.get("page", 1), chunk_id=c.get("chunk_id")) for c in s.get("citations", []) if c.get("chunk_id") in allowed_ids]
        final_steps.append(ScenarioStepOutput(step_number=s.get("step_number", i + 1), text=text, citations=cites))

    return ScenarioQAResponseOutput(
        doc_id=doc_id, question=question, scenario_type=scenario_type,
        steps=final_steps, confidence="high" if len(final_steps) >= 3 else "medium", disclaimer=QA_RESPONSE_DISCLAIMER
    )


def _handle_section_detail(doc_id: str, question: str, section_name: str) -> QAResponseOutput:
    chunks = retrieve_for_section(doc_id, section_name)
    summary: SectionSummaryWithConfidence = summarize_section(section_name, chunks, detail_level="detailed")
    answer_text = f"Detailed overview of {section_name}:\n" + "\n".join([f"- {b.text}" for b in summary.bullets])
    all_citations = []
    seen_chunks = set()
    for b in summary.bullets:
        for c in b.citations:
            if c.chunk_id not in seen_chunks:
                all_citations.append(c)
                seen_chunks.add(c.chunk_id)

    return QAResponseOutput(
        doc_id=doc_id, question=question, answer=answer_text, answer_type="section_detail",
        citations=all_citations, confidence=summary.confidence, disclaimer=QA_RESPONSE_DISCLAIMER,
        validation_issues=summary.validation_issues
    )


def route_question(doc_id: str, question: str) -> Any:
    q_lower = (question or "").lower()
    if any(phrase in q_lower for phrase in SCENARIO_TRIGGER_PHRASES):
        scenario_type = "ER" if "er" in q_lower or "emergency" in q_lower else "General"
        return ask_scenario(doc_id, question, scenario_type=scenario_type)
    if any(re.search(p, q_lower) for p in DETAIL_INTENT_PATTERNS):
        for section in CORE_SECTIONS:
            if section.lower() in q_lower:
                return _handle_section_detail(doc_id, question, section)
    return ask(doc_id, question)