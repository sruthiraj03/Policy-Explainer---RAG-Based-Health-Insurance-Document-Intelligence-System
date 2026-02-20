"""Tests for grounded Q&A engine."""

from unittest.mock import MagicMock, patch

import pytest

from backend.qa import ask, NOT_FOUND_ANSWER
from backend.schemas import QA_RESPONSE_DISCLAIMER, QAResponseOutput


def test_ask_empty_question_returns_not_found() -> None:
    out = ask("doc1", "")
    assert out.answer_type == "not_found"
    assert out.answer == NOT_FOUND_ANSWER
    assert out.citations == []
    assert out.disclaimer == QA_RESPONSE_DISCLAIMER
    assert out.doc_id == "doc1"


def test_ask_no_chunks_returns_not_found() -> None:
    with patch("backend.storage.query", return_value=[]):
        out = ask("doc1", "What is the deductible?")
    assert out.answer_type == "not_found"
    assert out.answer == NOT_FOUND_ANSWER
    assert out.confidence == "low"


def test_ask_returns_structured_output() -> None:
    chunks = [
        {"chunk_id": "c_5_0", "page_number": 5, "chunk_text": "The deductible is $500 per year."},
    ]
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '''{
        "answer_type": "answerable",
        "answer": "The deductible is $500 per year.",
        "citations": [{"page": 5, "chunk_id": "c_5_0"}]
    }'''
    with (
        patch("backend.qa.qa_engine.vector_query", return_value=chunks),
        patch("backend.qa.qa_engine.OpenAI") as mock_openai,
    ):
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        out = ask("doc1", "What is the deductible?")
    assert isinstance(out, QAResponseOutput)
    assert out.doc_id == "doc1"
    assert out.question == "What is the deductible?"
    assert out.answer_type == "answerable"
    assert out.disclaimer == QA_RESPONSE_DISCLAIMER
    assert len(out.citations) == 1
    assert out.citations[0].page == 5
    assert out.citations[0].chunk_id == "c_5_0"
    assert out.confidence in ("high", "medium", "low")


def test_ask_ambiguous_includes_clarification() -> None:
    chunks = [{"chunk_id": "c_1_0", "page_number": 1, "chunk_text": "Coverage varies."}]
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '''{
        "answer_type": "ambiguous",
        "answer": "",
        "citations": [],
        "clarification_question": "Which service or benefit are you asking about?"
    }'''
    with (
        patch("backend.storage.query", return_value=chunks),
        patch("backend.qa.OpenAI") as mock_openai,
    ):
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        out = ask("doc1", "What is covered?")
    assert out.answer_type == "ambiguous"
    assert out.citations == []
    assert out.confidence == "low"
    assert out.disclaimer == QA_RESPONSE_DISCLAIMER
