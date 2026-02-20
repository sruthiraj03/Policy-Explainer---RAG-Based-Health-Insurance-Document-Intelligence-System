"""Tests for core section constants (scaffolding)."""

import pytest

from backend.retrieval import CORE_SECTIONS


def test_core_sections_defined() -> None:
    """Core sections tuple is non-empty and contains expected names."""
    assert len(CORE_SECTIONS) == 6
    assert "Plan Snapshot" in CORE_SECTIONS
    assert "Exclusions & Limitations" in CORE_SECTIONS
