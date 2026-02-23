"""
Evaluation Module: The Judge.
Handles validation, confidence scoring, and quantitative metrics for:
1. Faithfulness (Groundedness)
2. Completeness (Coverage)
3. Simplicity (Readability)
"""

import json
import re
from pathlib import Path
from typing import Any

from backend import storage
from backend.schemas import SectionSummaryWithConfidence, BulletWithCitations

# --- Constants & Patterns ---

USER_FACING_CITATION_FORMAT = "(p. {page})"
SENTENCE_PATTERN = re.compile(r"[.!?]+")

# Weighting for Completeness Score based on Business/Policy Importance
SECTION_WEIGHTS: dict[str, float] = {
    "Plan Snapshot": 0.05,
    "Cost Summary": 0.35,
    "Summary of Covered Services": 0.30,
    "Administrative Conditions": 0.15,
    "Exclusions & Limitations": 0.10,
    "Claims, Appeals & Member Rights": 0.05,
}


# --- Internal Helpers ---

def _count_sentences(text: str) -> int:
    """Calculates sentence count to monitor verbosity."""
    if not text or not text.strip():
        return 0
    return len([p for p in SENTENCE_PATTERN.split(text.strip()) if p.strip()])


def _section_addressed(sec: SectionSummaryWithConfidence) -> bool:
    """
    Determines if a section is 'addressed'.
    Considered addressed if the LLM correctly identified it's missing (present=False)
    OR if it provided bullets with valid citations.
    """
    if not sec.present:
        return True
    if not sec.bullets:
        return False
    return any(b.citations for b in sec.bullets)


def _normalize_tokens(text: str) -> set[str]:
    """Basic tokenization for overlap analysis."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _extract_numbers(text: str) -> set[str]:
    """Extracts numeric values for fact-checking deductibles/copays."""
    return set(re.findall(r"\d+\.?\d*", text))


def _chunk_supports_bullet(bullet_text: str, chunk: Any, min_overlap: float = 0.15) -> bool:
    """Verifies if a specific text chunk supports a summary bullet."""
    bullet_tokens = _normalize_tokens(bullet_text)
    # Using getattr for flexible compatibility with different storage backends
    chunk_text = getattr(chunk, "chunk_text", "") or ""
    chunk_tokens = _normalize_tokens(chunk_text)

    if not bullet_tokens:
        return True

    # Check 1: Semantic keyword overlap
    if len(bullet_tokens & chunk_tokens) / len(bullet_tokens) >= min_overlap:
        return True

    # Check 2: Hard numeric fact checking (crucial for MSBA accuracy)
    bullet_nums = _extract_numbers(bullet_text)
    chunk_nums = _extract_numbers(chunk_text)
    if bullet_nums and bullet_nums <= chunk_nums:
        return True

    return False


# --- Validation Logic ---
def validate_section_summary(
        section_out: SectionSummaryWithConfidence,
        detail_level: str = "standard",
) -> tuple[bool, list[str]]:
    """Checks for citation errors and bullet count violations."""
    issues: list[str] = []

    # If the section isn't present, there are no issues to report
    if not section_out.present:
        return (True, [])

    bullets = section_out.bullets or []

    # Set the strictness for bullet counts based on detail level
    min_b, max_b = (3, 6) if detail_level == "standard" else (6, 12)

    if len(bullets) > max_b:
        issues.append(f"bullet_count_high: {len(bullets)} bullets (max {max_b})")

    if len(bullets) < min_b and len(bullets) > 0:
        issues.append(f"bullet_count_low: {len(bullets)} bullets (min {min_b})")

    for i, b in enumerate(bullets):
        # Every summary point MUST have at least one citation to be 'Faithful'
        if not b.citations:
            issues.append(f"bullet_{i + 1}_missing_citations")

        for c in b.citations:
            # Check for hallucinated page numbers
            if c.page <= 0:
                issues.append(f"bullet_{i + 1}_invalid_page_number: {c.page}")

            # Check for hallucinated chunk IDs (must follow our c_X_X format)
            if not c.chunk_id or not str(c.chunk_id).startswith("c_"):
                issues.append(f"bullet_{i + 1}_invalid_chunk_id: {c.chunk_id}")

    # A section is valid only if the issues list is empty
    return (len(issues) == 0, issues)


# --- Confidence Scoring ---

def confidence_for_section(
        section_out: SectionSummaryWithConfidence
) -> str:
    """Heuristic to determine 'Trustworthiness' of a section summary."""
    # We now pull the issues directly from the object we built in summarization.py
    issues = section_out.validation_issues or []

    # If the section isn't there or has no bullets, it's a low-confidence state
    if not section_out.present or not section_out.bullets:
        return "low"

    total_bullets = len(section_out.bullets)
    bullets_with_citations = sum(1 for b in section_out.bullets if b.citations)

    # Check for critical issues like hallucinated chunk IDs
    if issues:
        critical = [i for i in issues if any(x in i.lower() for x in ["invalid", "missing"])]
        if critical:
            return "low"
        return "medium"

    # If every bullet has a source, it's high confidence!
    if bullets_with_citations >= total_bullets:
        return "high"

    return "medium"

# --- Main Metrics Computation ---

def compute_faithfulness(doc_id: str, base_path: Path | None = None) -> dict[str, Any]:
    """Measures hallucination rate by verifying every cited bullet against its chunk."""
    try:
        summary = storage.load_policy_summary(doc_id, base_path)
        chunks_list = storage.load_chunks(doc_id, base_path)
    except FileNotFoundError:
        return {"error": "data_missing", "faithfulness_score": 0.0}

    chunks_by_id = {c.chunk_id: c for c in chunks_list}
    total_units = supported_units = 0

    for sec in summary.sections:
        if not sec.present or not sec.bullets:
            continue
        for b in sec.bullets:
            total_units += 1
            is_supported = False
            for cit in b.citations:
                ch = chunks_by_id.get(cit.chunk_id)
                if ch and _chunk_supports_bullet(b.text, ch):
                    is_supported = True
                    break
            if is_supported:
                supported_units += 1

    return {
        "doc_id": doc_id,
        "faithfulness_score": round(supported_units / (total_units or 1), 4),
        "total_units": total_units
    }


def compute_completeness(doc_id: str, base_path: Path | None = None) -> dict[str, Any]:
    """Calculates weighted coverage based on policy sections."""
    try:
        summary = storage.load_policy_summary(doc_id, base_path)
    except FileNotFoundError:
        return {"doc_id": doc_id, "error": "summary_missing", "completeness_score": 0.0}

    section_scores = {}
    weighted_sum = 0.0
    total_weight = sum(SECTION_WEIGHTS.values())

    for sec in summary.sections:
        name = sec.section_name
        weight = SECTION_WEIGHTS.get(name, 0.0)
        addressed = _section_addressed(sec)
        section_scores[name] = 1.0 if addressed else 0.0
        weighted_sum += weight * section_scores[name]

    return {
        "doc_id": doc_id,
        "completeness_score": round(weighted_sum / (total_weight or 1), 4),
        "section_scores": section_scores,
    }


def run_all_evaluations(doc_id: str, base_path: Path | None = None) -> dict[str, Any]:
    """Runner for the full analytics suite."""
    f_rep = compute_faithfulness(doc_id, base_path)
    c_rep = compute_completeness(doc_id, base_path)

    return {
        "doc_id": doc_id,
        "faithfulness": f_rep.get("faithfulness_score", 0.0),
        "completeness": c_rep.get("completeness_score", 0.0)
    }

# -- qa validation set --
def validate_qa_response(
        response_json: dict[str, Any],
        *,
        valid_page_numbers: set[int] | None = None,
) -> tuple[bool, list[str], str]:
    """Validates a QA response for disclaimer presence and citation accuracy."""
    issues = []
    answer = response_json.get("answer", "")
    citations = response_json.get("citations", [])

    if not response_json.get("disclaimer"):
        issues.append("disclaimer_required")

    if valid_page_numbers:
        for c in citations:
            p = c.get("page")
            if isinstance(p, int) and p not in valid_page_numbers:
                issues.append(f"invalid_page_citation:{p}")

    # Return (is_valid, issues, display_text)
    return len(issues) == 0, issues, answer


def confidence_for_qa(
        answer_type: str,
        citation_count: int,
        *,
        validation_issues: list[str] | None = None,
        retrieval_chunk_count: int = 0,
) -> str:
    """Heuristic to determine QA confidence (high/medium/low)."""
    issues = validation_issues or []
    if answer_type == "not_found" or retrieval_chunk_count == 0:
        return "low"
    if any("invalid" in i for i in issues):
        return "low"
    if citation_count >= 2 and retrieval_chunk_count >= 3:
        return "high"
    return "medium"