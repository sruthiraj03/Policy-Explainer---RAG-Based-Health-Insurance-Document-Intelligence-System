"""Tests for section deep-dive capability."""

from unittest.mock import patch

import pytest

from backend.qa import ask_section_detail, detect_section_detail_intent, route_question
from backend.schemas import BulletWithCitations, Citation, SectionDetailQAResponseOutput, SectionSummaryOutput


def test_detect_section_detail_intent_more_detail_about() -> None:
    assert detect_section_detail_intent("Give me more detail about Cost Summary") == "Cost Summary"
    assert detect_section_detail_intent("More detail about Exclusions & Limitations") == "Exclusions & Limitations"


def test_detect_section_detail_intent_explain_in_more_detail() -> None:
    assert detect_section_detail_intent("Explain Administrative Conditions in more detail") == "Administrative Conditions"


def test_detect_section_detail_intent_deeper_summary() -> None:
    assert detect_section_detail_intent("Provide a deeper summary of Exclusions") == "Exclusions & Limitations"


def test_detect_section_detail_intent_no_match() -> None:
    assert detect_section_detail_intent("What is the deductible?") is None
    assert detect_section_detail_intent("") is None


def test_route_question_returns_none_for_normal_question() -> None:
    assert route_question("doc1", "What is covered?") is None


def test_ask_section_detail_returns_structured_output() -> None:
    stub_summary = SectionSummaryOutput(
        section_name="Cost Summary",
        present=True,
        bullets=[
            BulletWithCitations(text="Deductible is $500.", citations=[Citation(page=5, chunk_id="c_5_0")]),
            BulletWithCitations(text="Copay $25.", citations=[Citation(page=5, chunk_id="c_5_1")]),
        ],
        not_found_message=None,
    )
    with patch("backend.retrieval.retrieve_for_section", return_value=[{"chunk_id": "c_5_0", "page_number": 5, "chunk_text": "Deductible $500."}]):
        with patch("backend.summarization.summarize_section", return_value=stub_summary):
            out = ask_section_detail("doc1", "More detail about Cost Summary", "Cost Summary")
    assert isinstance(out, SectionDetailQAResponseOutput)
    assert out.answer_type == "section_detail"
    assert out.section_id == "Cost Summary"
    assert len(out.bullets) == 2
    assert len(out.citations) == 1  # unique pages
    assert out.citations[0].page == 5
    assert out.disclaimer


def test_ask_section_detail_not_found() -> None:
    stub_summary = SectionSummaryOutput(
        section_name="Cost Summary",
        present=False,
        bullets=[],
        not_found_message="Not found in this document.",
    )
    with patch("backend.retrieval.retrieve_for_section", return_value=[]):
        with patch("backend.summarization.summarize_section", return_value=stub_summary):
            out = ask_section_detail("doc1", "More detail about Cost Summary", "Cost Summary")
    assert out.answer_type == "section_detail"
    assert out.bullets == []
    assert out.citations == []
    assert "Not found" in out.answer or "not found" in out.answer.lower()
