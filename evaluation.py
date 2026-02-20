"""Evaluation: validation, confidence, faithfulness, completeness, simplicity, runner."""

import json
import re
from pathlib import Path
from typing import Any

from backend import storage
from backend.schemas import SectionSummaryOutput

# --- Validation ---
USER_FACING_CITATION_FORMAT = "(p. {page})"
SENTENCE_PATTERN = re.compile(r"[.!?]+")


def _count_sentences(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len([p for p in SENTENCE_PATTERN.split(text.strip()) if p.strip()])


def _answerable_has_factual_content(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a or "not found in this document" in a:
        return False
    return True


def validate_qa_response(
    response_json: dict[str, Any],
    *,
    valid_page_numbers: set[int] | None = None,
) -> tuple[bool, list[str], str]:
    issues: list[str] = []
    answer = (response_json.get("answer") or "").strip()
    answer_type = response_json.get("answer_type") or "not_found"
    citations = response_json.get("citations") or []
    disclaimer = (response_json.get("disclaimer") or "").strip()
    if not disclaimer:
        issues.append("disclaimer_required")
    if answer_type == "not_found":
        return (len(issues) == 0, issues, answer)
    if answer_type == "ambiguous":
        return (len(issues) == 0, issues, answer)
    if answer_type == "conflict":
        answer_display = answer + " " + " ".join(USER_FACING_CITATION_FORMAT.format(page=p) for p in sorted({c.get("page") for c in citations if isinstance(c.get("page"), int)})) if citations else answer
        if valid_page_numbers:
            for c in citations:
                p = c.get("page") if isinstance(c, dict) else None
                if isinstance(p, int) and p not in valid_page_numbers:
                    issues.append(f"invalid_page_citation:{p}")
        return (len(issues) == 0, issues, answer_display)
    if answer_type == "answerable":
        if _answerable_has_factual_content(answer) and not citations:
            issues.append("answerable_but_no_citations")
        if _count_sentences(answer) > 6:
            issues.append("sentence_count_exceeds_6")
        if valid_page_numbers:
            for c in citations:
                p = c.get("page") if isinstance(c, dict) else None
                if isinstance(p, int) and p not in valid_page_numbers:
                    issues.append(f"invalid_page_citation:{p}")
        answer_display = answer + " " + " ".join(USER_FACING_CITATION_FORMAT.format(page=p) for p in sorted({c.get("page") for c in citations if isinstance(c.get("page"), int)})) if citations else answer
        return (len(issues) == 0, issues, answer_display)
    return (False, issues + ["unknown_answer_type"], answer)


def validate_section_summary(
    section_out: SectionSummaryOutput,
    *,
    valid_chunk_ids: set[str] | None = None,
    valid_page_numbers: set[int] | None = None,
    detail_level: str = "standard",
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not section_out.present:
        return (True, [])
    bullets = section_out.bullets or []
    min_b, max_b = (4, 6) if detail_level == "standard" else (8, 12)
    if len(bullets) > max_b:
        issues.append(f"bullet_count_exceeds_{max_b}")
    for i, b in enumerate(bullets):
        if not b.citations:
            issues.append(f"bullet_{i+1}_missing_citations")
        for c in b.citations:
            if valid_page_numbers is not None and c.page not in valid_page_numbers:
                issues.append(f"bullet_{i+1}_invalid_page:{c.page}")
            if valid_chunk_ids is not None and c.chunk_id not in valid_chunk_ids:
                issues.append(f"bullet_{i+1}_invalid_chunk_id:{c.chunk_id}")
    return (len(issues) == 0, issues)


# --- Confidence ---

def confidence_for_section(
    section_out: SectionSummaryOutput,
    *,
    validation_issues: list[str] | None = None,
    retrieval_chunk_count: int = 0,
) -> str:
    issues = validation_issues or []
    if not section_out.present or not section_out.bullets or retrieval_chunk_count == 0:
        return "low"
    bullets_with_citations = sum(1 for b in section_out.bullets if b.citations)
    total_bullets = len(section_out.bullets)
    complete_citations = total_bullets > 0 and bullets_with_citations >= total_bullets
    if issues:
        critical = [i for i in issues if "invalid_page" in i or "invalid_chunk" in i or "missing_citations" in i]
        if critical:
            return "low"
        return "medium"
    if complete_citations and retrieval_chunk_count >= 3:
        return "high"
    if bullets_with_citations >= total_bullets:
        return "medium"
    return "low"


def confidence_for_qa(
    answer_type: str,
    citation_count: int,
    *,
    validation_issues: list[str] | None = None,
    retrieval_chunk_count: int = 0,
    retrieval_strong: bool = False,
) -> str:
    issues = validation_issues or []
    if answer_type in ("not_found", "ambiguous", "conflict") or answer_type != "answerable":
        return "low"
    if "answerable_but_no_citations" in issues or "invalid_page_citation" in issues or citation_count == 0 or retrieval_chunk_count == 0:
        return "low"
    if issues:
        return "medium"
    if citation_count >= 2 and (retrieval_strong or retrieval_chunk_count >= 3):
        return "high"
    if citation_count >= 1:
        return "medium"
    return "low"


# --- Faithfulness ---

def _normalize_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+\.?\d*", text))


def _chunk_supports_bullet(bullet_text: str, chunk: Any, min_overlap: float = 0.15) -> bool:
    bullet_tokens = _normalize_tokens(bullet_text)
    chunk_tokens = _normalize_tokens(getattr(chunk, "chunk_text", None) or "")
    if not bullet_tokens:
        return True
    if len(bullet_tokens & chunk_tokens) / len(bullet_tokens) >= min_overlap:
        return True
    bullet_nums = _extract_numbers(bullet_text)
    chunk_nums = _extract_numbers(getattr(chunk, "chunk_text", None) or "")
    if bullet_nums and bullet_nums <= chunk_nums:
        return True
    return False


def compute_faithfulness(doc_id: str, base_path: Path | None = None) -> dict[str, Any]:
    try:
        summary = storage.load_policy_summary(doc_id, base_path)
    except FileNotFoundError:
        return {"doc_id": doc_id, "error": "policy_summary_not_found", "faithfulness_score": 0.0, "hallucination_rate": 0.0, "contradiction_rate": 0.0, "total_units": 0, "supported_units": 0}
    try:
        chunks_list = storage.load_chunks(doc_id, base_path)
    except FileNotFoundError:
        return {"doc_id": doc_id, "error": "chunks_not_found", "faithfulness_score": 0.0, "hallucination_rate": 0.0, "contradiction_rate": 0.0, "total_units": 0, "supported_units": 0}
    chunks_by_id = {c.chunk_id: c for c in chunks_list}
    total_units = supported_units = unsupported = 0
    unit_details: list[dict[str, Any]] = []
    for sec in summary.sections:
        if not sec.present or not sec.bullets:
            continue
        for b in sec.bullets:
            total_units += 1
            if not b.citations:
                unsupported += 1
                unit_details.append({"section": sec.section_name, "text_preview": b.text[:80], "supported": False, "reason": "no_citations"})
                continue
            supported = True
            reason = "supported"
            for cit in b.citations:
                ch = chunks_by_id.get(cit.chunk_id)
                if not ch:
                    supported, reason = False, f"chunk_missing:{cit.chunk_id}"
                    break
                if not _chunk_supports_bullet(b.text, ch):
                    supported, reason = False, "low_overlap"
                    break
            if supported:
                supported_units += 1
            else:
                unsupported += 1
            unit_details.append({"section": sec.section_name, "text_preview": b.text[:80], "supported": supported, "reason": reason})
    total_units = total_units or 1
    return {
        "doc_id": doc_id,
        "faithfulness_score": round(supported_units / total_units, 4),
        "hallucination_rate": round(unsupported / total_units, 4),
        "contradiction_rate": 0.0,
        "total_units": total_units,
        "supported_units": supported_units,
        "unit_details": unit_details,
    }


def save_faithfulness_report(report: dict[str, Any], doc_id: str, base_path: Path | None = None) -> Path:
    doc_dir = storage.get_document_dir(doc_id, base_path)
    doc_dir.mkdir(parents=True, exist_ok=True)
    path = doc_dir / storage.FAITHFULNESS_REPORT_FILENAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


# --- Completeness ---
SECTION_WEIGHTS: dict[str, float] = {
    "Plan Snapshot": 0.05,
    "Cost Summary": 0.35,
    "Covered Services": 0.30,
    "Administrative Conditions": 0.15,
    "Exclusions & Limitations": 0.10,
    "Claims/Appeals & Member Rights": 0.05,
}


def _section_addressed(sec: Any) -> bool:
    if not sec.present:
        return True
    if not sec.bullets:
        return False
    return any(b.citations for b in sec.bullets)


def compute_completeness(doc_id: str, base_path: Path | None = None) -> dict[str, Any]:
    try:
        summary = storage.load_policy_summary(doc_id, base_path)
    except FileNotFoundError:
        return {"doc_id": doc_id, "error": "policy_summary_not_found", "completeness_score": 0.0, "section_scores": {}, "weights": SECTION_WEIGHTS}
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
        "completeness_score": round(weighted_sum / total_weight if total_weight else 0.0, 4),
        "section_scores": section_scores,
        "weights": SECTION_WEIGHTS,
    }


def save_completeness_report(report: dict[str, Any], doc_id: str, base_path: Path | None = None) -> Path:
    doc_dir = storage.get_document_dir(doc_id, base_path)
    doc_dir.mkdir(parents=True, exist_ok=True)
    path = doc_dir / storage.COMPLETENESS_REPORT_FILENAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


# --- Simplicity ---
JARGON_TERMS_PATH = Path(__file__).resolve().parent.parent / "schema" / "jargon_terms.json"


def _load_jargon_terms(path: Path | None = None) -> set[str]:
    p = path or JARGON_TERMS_PATH
    if not p.exists():
        return set()
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(t).lower() for t in data} if isinstance(data, list) else set()
    except Exception:
        return set()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()] if text and text.strip() else []


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text)


def _approx_syllables(word: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", word.lower()))) if word else 0


def _flesch_reading_ease(text: str) -> float:
    sentences, words = _sentences(text), _words(text)
    if not sentences or not words:
        return 0.0
    return 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (sum(_approx_syllables(w) for w in words) / len(words))


def _jargon_rate(text: str, jargon: set[str]) -> float:
    tokens = set(w.lower() for w in _words(text))
    return (sum(1 for t in tokens if t in jargon) / len(tokens)) if tokens else 0.0


def _avg_words_per_sentence(text: str) -> float:
    sents = _sentences(text)
    return sum(len(_words(s)) for s in sents) / len(sents) if sents else 0.0


def compute_simplicity(doc_id: str, base_path: Path | None = None, jargon_path: Path | None = None) -> dict[str, Any]:
    err = {"doc_id": doc_id, "simplicity_score": 0.0, "sentence_length_original": 0.0, "sentence_length_summary": 0.0, "jargon_rate_original": 0.0, "jargon_rate_summary": 0.0, "flesch_original": 0.0, "flesch_summary": 0.0}
    try:
        pages = storage.load_extracted_pages(doc_id, base_path)
    except FileNotFoundError:
        return {**err, "error": "pages_not_found"}
    try:
        summary = storage.load_policy_summary(doc_id, base_path)
    except FileNotFoundError:
        return {**err, "error": "policy_summary_not_found"}
    original_text = " ".join(p.text or "" for p in pages)
    summary_text = " ".join(b.text for sec in summary.sections for b in sec.bullets)
    jargon = _load_jargon_terms(jargon_path)
    avg_orig = _avg_words_per_sentence(original_text)
    avg_sum = _avg_words_per_sentence(summary_text)
    sentence_reduction = max(0.0, min(1.0, (avg_orig - avg_sum) / avg_orig if avg_orig > 0 else 0.0))
    jargon_orig = _jargon_rate(original_text, jargon)
    jargon_sum = _jargon_rate(summary_text, jargon)
    jargon_reduction = max(0.0, min(1.0, (jargon_orig - jargon_sum) / jargon_orig if jargon_orig > 0 else 0.0))
    flesch_orig = _flesch_reading_ease(original_text)
    flesch_sum = _flesch_reading_ease(summary_text)
    flesch_improvement = max(0.0, min(1.0, (flesch_sum - flesch_orig) / 100.0 if flesch_orig <= 100 else 0.0))
    simplicity_score = round(max(0.0, min(1.0, (sentence_reduction + jargon_reduction + flesch_improvement) / 3.0)), 4)
    return {
        "doc_id": doc_id,
        "simplicity_score": simplicity_score,
        "sentence_length_original": round(avg_orig, 2),
        "sentence_length_summary": round(avg_sum, 2),
        "sentence_length_reduction": round(sentence_reduction, 4),
        "jargon_rate_original": round(jargon_orig, 4),
        "jargon_rate_summary": round(jargon_sum, 4),
        "jargon_reduction": round(jargon_reduction, 4),
        "flesch_original": round(flesch_orig, 2),
        "flesch_summary": round(flesch_sum, 2),
        "flesch_improvement": round(flesch_improvement, 4),
    }


def save_simplicity_report(report: dict[str, Any], doc_id: str, base_path: Path | None = None) -> Path:
    doc_dir = storage.get_document_dir(doc_id, base_path)
    doc_dir.mkdir(parents=True, exist_ok=True)
    path = doc_dir / storage.SIMPLICITY_REPORT_FILENAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


# --- Runner ---

def run_all(doc_id: str, base_path: Path | None = None) -> dict[str, Any]:
    faithfulness_report = compute_faithfulness(doc_id, base_path)
    completeness_report = compute_completeness(doc_id, base_path)
    simplicity_report = compute_simplicity(doc_id, base_path)
    save_faithfulness_report(faithfulness_report, doc_id, base_path)
    save_completeness_report(completeness_report, doc_id, base_path)
    save_simplicity_report(simplicity_report, doc_id, base_path)
    evaluation_report: dict[str, Any] = {
        "doc_id": doc_id,
        "faithfulness_score": faithfulness_report.get("faithfulness_score", 0.0),
        "completeness_score": completeness_report.get("completeness_score", 0.0),
        "simplicity_score": simplicity_report.get("simplicity_score", 0.0),
        "errors": [],
    }
    for name, rep in [("faithfulness", faithfulness_report), ("completeness", completeness_report), ("simplicity", simplicity_report)]:
        if rep.get("error"):
            evaluation_report["errors"].append(f"{name}:{rep['error']}")
    doc_dir = storage.get_document_dir(doc_id, base_path)
    doc_dir.mkdir(parents=True, exist_ok=True)
    with (doc_dir / storage.EVALUATION_REPORT_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
    return evaluation_report
