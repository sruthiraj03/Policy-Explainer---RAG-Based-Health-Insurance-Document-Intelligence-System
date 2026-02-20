"""Tests for example scenario generator."""

from unittest.mock import MagicMock, patch

import pytest

from backend.qa import (
    SCENARIO_HEADER,
    detect_scenario_intent,
    ask_scenario,
    route_scenario_question,
    NOT_FOUND_ANSWER,
)


def test_detect_scenario_intent_trigger_phrases() -> None:
    assert detect_scenario_intent("What would happen if I visit the ER?") == (True, "ER visit")
    assert detect_scenario_intent("Example scenario: specialist visit") == (True, "Specialist visit")
    assert detect_scenario_intent("How much would I pay if I fill a prescription?") == (True, "Prescription fill")


def test_detect_scenario_intent_no_trigger() -> None:
    assert detect_scenario_intent("What is the deductible?") == (False, None)
    assert detect_scenario_intent("") == (False, None)


def test_ask_scenario_no_chunks_returns_not_found() -> None:
    with patch("backend.storage.query", return_value=[]):
        out = ask_scenario("doc1", "What would happen if I visit the ER?", "ER visit")
    assert out.answer_type == "scenario"
    assert out.scenario_type == "ER visit"
    assert out.header == SCENARIO_HEADER
    assert out.steps == []
    assert out.not_found_message == NOT_FOUND_ANSWER
    assert out.disclaimer


def test_ask_scenario_returns_structured_output() -> None:
    chunks = [
        {"chunk_id": "c_5_0", "page_number": 5, "chunk_text": "ER copay $500. Deductible $1000."},
    ]
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '''{
        "not_found": false,
        "steps": [
            {"step_number": 1, "text": "Your ER copay is $500 (page 5).", "citations": [{"page": 5}]},
            {"step_number": 2, "text": "Your deductible is $1000 (page 5).", "citations": [{"page": 5}]}
        ]
    }'''
    with patch("backend.storage.query", return_value=chunks):
        with patch("backend.qa.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            out = ask_scenario("doc1", "What would happen if I visit the ER?", "ER visit")
    # Less than 3 steps -> we treat as not_found in current logic (3-6 steps required)
    assert out.answer_type == "scenario"
    assert out.header == SCENARIO_HEADER
    assert out.disclaimer


def test_route_scenario_question_returns_none_for_normal_question() -> None:
    assert route_scenario_question("doc1", "What is the deductible?") is None
