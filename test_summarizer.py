"""Tests for section summarization engine."""

from unittest.mock import MagicMock, patch

import pytest

from backend.schemas import NOT_FOUND_MESSAGE, SectionSummaryOutput
from backend.summarization import summarize_section


def test_summarize_section_no_chunks_returns_not_found() -> None:
    out = summarize_section("Cost Summary", [], detail_level="standard")
    assert isinstance(out, SectionSummaryOutput)
    assert out.section_name == "Cost Summary"
    assert out.present is False
    assert out.bullets == []
    assert out.not_found_message == NOT_FOUND_MESSAGE


def test_summarize_section_empty_chunk_text_returns_not_found() -> None:
    chunks = [{"chunk_id": "c_1_0", "page_number": 1, "chunk_text": ""}]
    out = summarize_section("Cost Summary", chunks, detail_level="standard")
    assert out.present is False
    assert out.not_found_message == NOT_FOUND_MESSAGE


def test_summarize_section_llm_returns_present_false() -> None:
    chunks = [
        {"chunk_id": "c_2_0", "page_number": 2, "chunk_text": "Some unrelated text."},
    ]
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"present": false, "bullets": []}'
    with patch("backend.summarization.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        out = summarize_section("Exclusions & Limitations", chunks, detail_level="standard")
    assert out.present is False
    assert out.not_found_message == NOT_FOUND_MESSAGE


def test_summarize_section_llm_returns_bullets_with_citations() -> None:
    chunks = [
        {"chunk_id": "c_5_0", "page_number": 5, "chunk_text": "Deductible is $500 per year."},
    ]
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '''{
        "present": true,
        "bullets": [
            {"text": "Deductible is $500 per year.", "citations": [{"page": 5, "chunk_id": "c_5_0"}]}
        ]
    }'''
    with patch("backend.summarization.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        out = summarize_section("Cost Summary", chunks, detail_level="standard")
    assert out.present is True
    assert out.not_found_message is None
    assert len(out.bullets) == 1
    assert out.bullets[0].text == "Deductible is $500 per year."
    assert len(out.bullets[0].citations) == 1
    assert out.bullets[0].citations[0].page == 5
    assert out.bullets[0].citations[0].chunk_id == "c_5_0"


def test_summarize_section_strips_invalid_citations() -> None:
    """Citations referencing chunk_ids not in context are dropped."""
    chunks = [{"chunk_id": "c_3_0", "page_number": 3, "chunk_text": "Copay $25."}]
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '''{
        "present": true,
        "bullets": [
            {"text": "Copay is $25.", "citations": [{"page": 3, "chunk_id": "c_3_0"}, {"page": 99, "chunk_id": "c_99_0"}]}
        ]
    }'''
    with patch("backend.summarization.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        out = summarize_section("Cost Summary", chunks, detail_level="standard")
    assert out.present is True
    assert len(out.bullets[0].citations) == 1
    assert out.bullets[0].citations[0].chunk_id == "c_3_0"
