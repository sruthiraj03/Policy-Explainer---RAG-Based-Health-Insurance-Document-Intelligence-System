"""Tests for evaluation modules and runner."""

from pathlib import Path
from unittest.mock import patch

import pytest

from backend.evaluation import (
    compute_completeness,
    compute_faithfulness,
    compute_simplicity,
    run_all,
    SECTION_WEIGHTS,
)


def test_run_all_graceful_when_summary_missing(tmp_path: Path) -> None:
    """run_all does not raise when policy summary is missing; returns report with errors."""
    doc_id = "nonexistent-doc"
    (tmp_path / doc_id).mkdir(parents=True)
    report = run_all(doc_id, base_path=tmp_path)
    assert "doc_id" in report
    assert "errors" in report
    assert report["faithfulness_score"] == 0.0
    assert report["completeness_score"] == 0.0
    assert report["simplicity_score"] == 0.0


def test_compute_faithfulness_missing_summary_returns_error_dict(tmp_path: Path) -> None:
    out = compute_faithfulness("no-such-doc", base_path=tmp_path)
    assert "error" in out
    assert out["faithfulness_score"] == 0.0


def test_compute_completeness_section_weights() -> None:
    assert SECTION_WEIGHTS["Cost Summary"] == 0.35
    assert sum(SECTION_WEIGHTS.values()) == pytest.approx(1.0)


def test_compute_simplicity_missing_pages_returns_error_dict(tmp_path: Path) -> None:
    out = compute_simplicity("no-such-doc", base_path=tmp_path)
    assert "error" in out
    assert out["simplicity_score"] == 0.0
