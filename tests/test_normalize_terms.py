"""Tests for terminology normalization."""

import pytest

from backend.utils import (
    load_terminology_map,
    normalize_text,
)


def test_load_terminology_map_default() -> None:
    """Default path loads schema/terminology_map.json."""
    m = load_terminology_map()
    assert isinstance(m, dict)
    assert "out-of-pocket maximum" in m
    assert "oop max" in m["out-of-pocket maximum"]


def test_normalize_text_canonical_terms() -> None:
    """Synonyms are replaced with canonical form."""
    text = "Your OOP max is $5000. The annual deductible applies first."
    out = normalize_text(text)
    assert "out-of-pocket maximum" in out
    assert "deductible" in out
    assert "coinsurance" in normalize_text("Co-insurance is 20%.")


def test_normalize_text_whole_phrase_only() -> None:
    """Partial words are not replaced (e.g. 'network' in 'networking')."""
    out = normalize_text("Networking is important.", {"in-network": ["network"]})
    assert out == "Networking is important."


def test_normalize_text_quoted_unchanged() -> None:
    """Quoted policy snippets are not modified."""
    text = 'The policy says "oop max" and annual deductible.'
    out = normalize_text(text)
    assert '"oop max"' in out
    assert "deductible" in out


def test_normalize_text_empty_map_unchanged() -> None:
    """Empty terminology map returns text unchanged."""
    assert normalize_text("OOP max here.", {}) == "OOP max here."


def test_normalize_text_custom_map() -> None:
    """Custom terminology_map is used when provided."""
    m = {"canonical": ["synonym"]}
    assert normalize_text("Use synonym here.", m) == "Use canonical here."
