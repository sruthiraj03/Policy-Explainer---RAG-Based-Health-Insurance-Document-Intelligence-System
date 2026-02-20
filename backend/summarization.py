"""Summarization: section summarizer and full pipeline orchestration."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from backend import storage
from backend.config import get_settings
from backend.evaluation import confidence_for_section, validate_section_summary
from backend.retrieval import CORE_SECTIONS, retrieve_for_section
from backend.schemas import (
    BulletWithCitations,
    Citation,
    DEFAULT_DISCLAIMER,
    DetailLevel,
    DocMetadata,
    NOT_FOUND_MESSAGE,
    PolicySummaryOutput,
    SectionSummaryOutput,
    SectionSummaryWithConfidence,
)
from backend.utils import load_terminology_map, normalize_text

STANDARD_MAX_BULLETS = 6
STANDARD_MIN_BULLETS = 4
DETAILED_MAX_BULLETS = 12
DETAILED_MIN_BULLETS = 8


def _build_context(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return ""
    parts = []
    for c in chunks:
        page = c.get("page_number", 0)
        cid = c.get("chunk_id", "")
        text = (c.get("chunk_text") or "").strip()
        parts.append(f"---\nChunk {cid} (page {page}):\n{text}\n")
    return "\n".join(parts).strip()


def _build_system_prompt() -> str:
    return """You are a policy document summarizer. Your task is to summarize ONLY the provided document excerpts. Rules:
- Use ONLY information from the chunks provided below. Do not add any information from outside the document.
- Never guess or infer facts not stated in the chunks.
- Output valid JSON only: {"present": true|false, "bullets": [{"text": "...", "citations": [{"page": N, "chunk_id": "c_X_Y"}]}]}.
- Every bullet MUST have at least one citation. Use only page numbers and chunk_ids that appear in the provided chunks.
- If the chunks do not contain relevant information for this section, set "present": false and "bullets": [].
- Write in plain English. Use bullets (short phrases or sentences), not long paragraphs.
- Citations must reference exact chunk_id and page from the context (e.g. c_5_0, page 5)."""


def _build_user_prompt(section_name: str, context: str, detail_level: DetailLevel) -> str:
    min_b, max_b = (DETAILED_MIN_BULLETS, DETAILED_MAX_BULLETS) if detail_level == "detailed" else (STANDARD_MIN_BULLETS, STANDARD_MAX_BULLETS)
    return f"""Section to summarize: "{section_name}"

Document excerpts (use ONLY these; cite each bullet with page and chunk_id from below):
{context}

Produce between {min_b} and {max_b} bullets. Each bullet: "text" (plain English) and "citations" (list of {{"page": <int>, "chunk_id": "<id>"}} from the excerpts above).
If there is no relevant content for this section in the excerpts, respond with: {{"present": false, "bullets": []}}
Output only the JSON object, no markdown or explanation."""


def _parse_llm_json(raw: str) -> dict[str, Any] | None:
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


def _validate_citations(bullets: list[dict], allowed_chunk_ids: set[str]) -> list[BulletWithCitations]:
    out = []
    for b in bullets:
        text = b.get("text") or ""
        valid_cites = []
        for c in b.get("citations") or []:
            if not isinstance(c, dict):
                continue
            page, cid = c.get("page"), c.get("chunk_id")
            if isinstance(page, int) and page >= 1 and isinstance(cid, str) and cid and cid in allowed_chunk_ids:
                valid_cites.append(Citation(page=page, chunk_id=cid))
        out.append(BulletWithCitations(text=text, citations=valid_cites))
    return out


def _clamp_bullets(bullets: list[BulletWithCitations], detail_level: DetailLevel) -> list[BulletWithCitations]:
    max_b = DETAILED_MAX_BULLETS if detail_level == "detailed" else STANDARD_MAX_BULLETS
    return bullets[:max_b]


def summarize_section(
    section_name: str,
    chunks: list[dict[str, Any]],
    detail_level: DetailLevel = "standard",
) -> SectionSummaryOutput:
    allowed_ids = {str(c.get("chunk_id")) for c in chunks if c.get("chunk_id")}
    if not chunks or not any((c.get("chunk_text") or "").strip() for c in chunks):
        return SectionSummaryOutput(section_name=section_name, present=False, bullets=[], not_found_message=NOT_FOUND_MESSAGE)
    context = _build_context(chunks)
    if not context.strip():
        return SectionSummaryOutput(section_name=section_name, present=False, bullets=[], not_found_message=NOT_FOUND_MESSAGE)
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": _build_user_prompt(section_name, context, detail_level)},
        ],
        temperature=0.1,
    )
    raw = (response.choices[0].message.content or "").strip()
    parsed = _parse_llm_json(raw)
    if not parsed:
        return SectionSummaryOutput(section_name=section_name, present=False, bullets=[], not_found_message=NOT_FOUND_MESSAGE)
    present = bool(parsed.get("present", True))
    bullets_raw = parsed.get("bullets") or []
    if not isinstance(bullets_raw, list):
        bullets_raw = []
    bullets = _validate_citations(bullets_raw, allowed_ids)
    bullets = [b for b in bullets if b.citations]
    bullets = _clamp_bullets(bullets, detail_level)
    term_map = load_terminology_map()
    bullets = [BulletWithCitations(text=normalize_text(b.text, term_map), citations=b.citations) for b in bullets]
    if not present or not bullets:
        return SectionSummaryOutput(section_name=section_name, present=False, bullets=[], not_found_message=NOT_FOUND_MESSAGE)
    return SectionSummaryOutput(section_name=section_name, present=True, bullets=bullets, not_found_message=None)


def extract_doc_metadata(doc_id: str, base_path: Path | None = None) -> DocMetadata:
    total_pages = 0
    try:
        total_pages = len(storage.load_extracted_pages(doc_id, base_path))
    except FileNotFoundError:
        pass
    return DocMetadata(doc_id=doc_id, generated_at=datetime.now(tz=timezone.utc).isoformat(), total_pages=total_pages, source_file=None)


def run_full_summary_pipeline(
    doc_id: str,
    detail_level: DetailLevel = "standard",
    base_path: Path | None = None,
    disclaimer: str | None = None,
) -> PolicySummaryOutput:
    metadata = extract_doc_metadata(doc_id, base_path)
    sections_with_confidence: list[SectionSummaryWithConfidence] = []
    for section_name in CORE_SECTIONS:
        chunks = retrieve_for_section(doc_id, section_name)
        section_out = summarize_section(section_name, chunks, detail_level=detail_level)
        valid_chunk_ids = {c.get("chunk_id") for c in chunks if c.get("chunk_id")}
        valid_page_numbers = {c.get("page_number") for c in chunks if c.get("page_number")}
        _, validation_issues = validate_section_summary(section_out, valid_chunk_ids=valid_chunk_ids, valid_page_numbers=valid_page_numbers, detail_level=detail_level)
        confidence = confidence_for_section(section_out, validation_issues=validation_issues, retrieval_chunk_count=len(chunks))
        sections_with_confidence.append(
            SectionSummaryWithConfidence(
                section_name=section_out.section_name,
                present=section_out.present,
                bullets=section_out.bullets,
                not_found_message=section_out.not_found_message,
                confidence=confidence,
                validation_issues=validation_issues,
            )
        )
    summary = PolicySummaryOutput(metadata=metadata, disclaimer=disclaimer or DEFAULT_DISCLAIMER, sections=sections_with_confidence)
    storage.save_policy_summary(summary, doc_id, base_path)
    return summary
