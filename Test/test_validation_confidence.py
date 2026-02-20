"""Tests for validation and deterministic confidence scoring."""

import pytest

from backend.evaluation import confidence_for_qa, confidence_for_section, validate_qa_response, validate_section_summary
from backend.schemas import BulletWithCitations, Citation, SectionSummaryOutput


def test_validate_qa_response_answerable_no_citations_blocked() -> None:
    r = {"answer": "The deductible is $500.", "answer_type": "answerable", "citations": [], "disclaimer": "D"}
    valid, issues, display = validate_qa_response(r)
    assert valid is False
    assert "answerable_but_no_citations" in issues


def test_validate_qa_response_not_found_may_omit_citations() -> None:
    r = {"answer": "Not found in this document.", "answer_type": "not_found", "citations": [], "disclaimer": "D"}
    valid, issues, display = validate_qa_response(r)
    assert valid is True
    assert display == "Not found in this document."


def test_validate_qa_response_disclaimer_required() -> None:
    r = {"answer": "Yes.", "answer_type": "answerable", "citations": [{"page": 5, "chunk_id": "c_5_0"}], "disclaimer": ""}
    valid, issues, _ = validate_qa_response(r)
    assert "disclaimer_required" in issues


def test_validate_qa_response_invalid_page_citation() -> None:
    r = {"answer": "It is $500.", "answer_type": "answerable", "citations": [{"page": 99, "chunk_id": "c_99_0"}], "disclaimer": "D"}
    valid, issues, _ = validate_qa_response(r, valid_page_numbers={5, 6})
    assert valid is False
    assert any("99" in i for i in issues)


def test_validate_section_summary_bullet_missing_citations() -> None:
    section = SectionSummaryOutput(
        section_name="Cost Summary",
        present=True,
        bullets=[
            BulletWithCitations(text="Deductible $500.", citations=[Citation(page=5, chunk_id="c_5_0")]),
            BulletWithCitations(text="No citation here.", citations=[]),
        ],
        not_found_message=None,
    )
    valid, issues = validate_section_summary(section)
    assert valid is False
    assert any("missing_citations" in i for i in issues)


def test_confidence_for_qa_answerable_with_citations_high() -> None:
    assert confidence_for_qa("answerable", 2, validation_issues=[], retrieval_chunk_count=5, retrieval_strong=True) == "high"


def test_confidence_for_qa_not_found_low() -> None:
    assert confidence_for_qa("not_found", 0) == "low"


def test_confidence_for_qa_answerable_no_citations_low() -> None:
    assert confidence_for_qa("answerable", 0, validation_issues=["answerable_but_no_citations"]) == "low"


def test_confidence_for_section_present_all_cited_high() -> None:
    section = SectionSummaryOutput(
        section_name="Cost Summary",
        present=True,
        bullets=[
            BulletWithCitations(text="A.", citations=[Citation(page=1, chunk_id="c_1_0")]),
            BulletWithCitations(text="B.", citations=[Citation(page=2, chunk_id="c_2_0")]),
        ],
        not_found_message=None,
    )
    assert confidence_for_section(section, validation_issues=[], retrieval_chunk_count=5) == "high"


def test_confidence_for_section_not_present_low() -> None:
    section = SectionSummaryOutput(section_name="X", present=False, bullets=[], not_found_message="Not found.")
    assert confidence_for_section(section) == "low"
