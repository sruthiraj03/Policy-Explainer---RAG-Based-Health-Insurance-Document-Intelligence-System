"""Tests for preventive care helper."""

from unittest.mock import patch

import pytest

from backend.qa import ask_preventive, FROM_POLICY_LABEL, GENERAL_GUIDANCE_LABEL, get_preventive_follow_up_questions, is_preventive_question
from backend.schemas import PreventiveQAResponseOutput, QA_RESPONSE_DISCLAIMER


def test_is_preventive_question_detects_keywords() -> None:
    assert is_preventive_question("Is the flu vaccine covered?") is True
    assert is_preventive_question("mammogram screening") is True
    assert is_preventive_question("annual physical exam") is True
    assert is_preventive_question("preventive wellness visit") is True
    assert is_preventive_question("colonoscopy screening") is True


def test_is_preventive_question_rejects_non_preventive() -> None:
    assert is_preventive_question("What is the deductible?") is False
    assert is_preventive_question("") is False
    assert is_preventive_question("   ") is False


def test_get_preventive_follow_up_questions_returns_5_to_8() -> None:
    qs = get_preventive_follow_up_questions()
    assert 5 <= len(qs) <= 8
    assert any("prior authorization" in q.lower() for q in qs)
    assert any("in-network" in q.lower() or "network" in q.lower() for q in qs)


def test_ask_preventive_includes_sections_and_disclaimer() -> None:
    from backend.schemas import QAResponseOutput
    stub = QAResponseOutput(
        doc_id="d1",
        question="Is mammogram covered?",
        answer="Not found in this document.",
        answer_type="not_found",
        citations=[],
        confidence="low",
        disclaimer=QA_RESPONSE_DISCLAIMER,
    )
    with patch("backend.qa.ask", return_value=stub):
        out = ask_preventive("d1", "Is mammogram covered?")
    assert isinstance(out, PreventiveQAResponseOutput)
    assert FROM_POLICY_LABEL in out.answer
    assert GENERAL_GUIDANCE_LABEL in out.answer
    assert out.disclaimer == QA_RESPONSE_DISCLAIMER
    assert len(out.follow_up_questions) >= 5
