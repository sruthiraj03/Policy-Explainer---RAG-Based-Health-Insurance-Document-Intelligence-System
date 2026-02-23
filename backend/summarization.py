"""
Summarization Module: The LLM Orchestrator.
This module takes retrieved text chunks and prompts an LLM to generate
structured summaries. It now returns the unified SectionSummaryWithConfidence
model to ensure compatibility with the updated schemas.py.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

from backend import storage
from backend.config import get_settings
from backend.evaluation import confidence_for_section, validate_section_summary
from backend.retrieval import CORE_SECTIONS, retrieve_for_section
from backend.schemas import (
    BulletWithCitations,
    Citation,
    DEFAULT_DISCLAIMER,
    DocMetadata,
    NOT_FOUND_MESSAGE,
    PolicySummaryOutput,
    SectionSummaryWithConfidence,  # Use the consolidated model
)
from backend.utils import load_terminology_map, normalize_text

# Define DetailLevel locally as it is a specific logic toggle for the pipeline
DetailLevel = Literal["standard", "detailed"]

# Consistency constants for bullet counts
STANDARD_MAX_BULLETS = 6
DETAILED_MAX_BULLETS = 12

def _build_context(chunks: list[dict[str, Any]]) -> str:
    """
    Formatter: Converts database chunks into a clear text block for the LLM.
    Labels each chunk with its ID and Page so the LLM has the 'keys' for citations.
    """
    if not chunks:
        return ""
    parts = []
    for c in chunks:
        page = c.get("page_number", 0)
        cid = c.get("chunk_id", "")
        text = (c.get("chunk_text") or "").strip()
        parts.append(f"---\nChunk {cid} (page {page}):\n{text}\n")
    return "\n".join(parts).strip()

def _parse_llm_json(raw: str) -> dict[str, Any] | None:
    """
    Robust JSON Extraction:
    Extracts JSON from the LLM's response even if it's wrapped in markdown backticks
    or preceded by conversational filler text.
    """
    s = raw.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.DOTALL)
    if m:
        s = m.group(1)
    else:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def summarize_section(
        section_name: str,
        chunks: list[dict[str, Any]],
        detail_level: DetailLevel = "standard",
) -> SectionSummaryWithConfidence:
    """
    The main logic for a single section summary.
    Instead of an intermediate output class, it maps directly to
    SectionSummaryWithConfidence to support the confidence scoring phase.
    """
    allowed_ids = {str(c.get("chunk_id")) for c in chunks if c.get("chunk_id")}

    # Pre-define the 'Not Found' state for sections with no data
    empty_res = SectionSummaryWithConfidence(
        section_name=section_name,
        present=False,
        bullets=[],
        not_found_message=NOT_FOUND_MESSAGE,
        confidence="low",
        validation_issues=["No relevant document chunks found."]
    )

    if not chunks:
        return empty_res

    context = _build_context(chunks)
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # --- THE FIX: Explicitly enforce JSON output and structure ---
    system_prompt = f"""You are a policy document summarizer. Use ONLY provided chunks.
    You MUST output your response as a valid JSON object with exactly these keys:
    - "present": boolean (true if info is found, false if not)
    - "bullets": a list of dicts, each with "text" (the summary point) and "citations" (a list of dicts with "chunk_id" and "page").
    """

    # Prompt logic uses strict rules to ensure the LLM stays grounded in the chunks
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize {section_name} using: {context}"},
            ],
            temperature=0.1,  # Low temperature to reduce hallucination
        )

        raw = (response.choices[0].message.content or "").strip()
        print(f"🤖 DEBUG SUMMARY LLM OUTPUT for {section_name}: {raw[:100]}...")  # Let's print a snippet
        parsed = _parse_llm_json(raw)

    except Exception as e:
        print(f"❌ DEBUG: OpenAI API Call Failed for {section_name}: {e}")
        return empty_res

    if not parsed or not parsed.get("present"):
        return empty_res

    # 1. Validate citations and Normalize terminology
    term_map = load_terminology_map()
    valid_bullets = []
    for b in parsed.get("bullets", []):
        text = normalize_text(b.get("text", ""), term_map)
        cites = [
            Citation(page=c.get('page', 0), chunk_id=c.get('chunk_id', ''))
            for c in b.get("citations", [])
            if str(c.get("chunk_id")) in allowed_ids
        ]
        # Only keep bullets that have at least one valid source in the doc
        if cites:
            valid_bullets.append(BulletWithCitations(text=text, citations=cites))

    # Use existing evaluation functions to check the summary quality
    _, issues = validate_section_summary(valid_bullets)
    conf = confidence_for_section(valid_bullets, issues, len(chunks))

    # 3. Construct the final confidence-wrapped object
    return SectionSummaryWithConfidence(
        section_name=section_name,
        present=True if valid_bullets else False,
        bullets=valid_bullets[:DETAILED_MAX_BULLETS if detail_level == "detailed" else STANDARD_MAX_BULLETS],
        not_found_message=None if valid_bullets else NOT_FOUND_MESSAGE,
        confidence=conf,
        validation_issues=issues
    )

def run_full_summary_pipeline(
    doc_id: str,
    detail_level: DetailLevel = "standard",
    base_path: Path | None = None,
) -> PolicySummaryOutput:
    """
    The orchestrator for the entire document summary.
    Loops through the standard policy sections, retrieves data, generates
    summaries, and saves the final PolicySummaryOutput object.
    """
    # Load metadata for the audit trail
    total_pages = len(storage.load_extracted_pages(doc_id, base_path))
    metadata = DocMetadata(
        doc_id=doc_id,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        total_pages=total_pages
    )

    # Generate summaries section by section
    final_sections = []
    for section_name in CORE_SECTIONS:
        # Fetch relevant chunks from the vector store
        chunks = retrieve_for_section(doc_id, section_name)
        # Process chunks into structured summary
        summary = summarize_section(section_name, chunks, detail_level)
        final_sections.append(summary)

    # Build the final comprehensive output
    full_output = PolicySummaryOutput(
        metadata=metadata,
        disclaimer=DEFAULT_DISCLAIMER,
        sections=final_sections
    )

    # Persist the summary for the frontend to display
    storage.save_policy_summary(full_output, doc_id, base_path)
    return full_output