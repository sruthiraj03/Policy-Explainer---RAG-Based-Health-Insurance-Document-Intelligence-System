"""Tests for section-specific retrieval."""

from unittest.mock import patch

import pytest

from backend.retrieval import CORE_SECTIONS, retrieve_for_section, SECTION_QUERIES


def test_section_queries_cover_core_sections() -> None:
    """Every core section has at least one query."""
    for section in CORE_SECTIONS:
        assert section in SECTION_QUERIES, f"Missing queries for section: {section}"
        assert len(SECTION_QUERIES[section]) >= 1


def test_retrieve_for_section_unknown_returns_empty() -> None:
    assert retrieve_for_section("doc1", "Unknown Section") == []


def test_retrieve_for_section_returns_chunks_with_metadata() -> None:
    """retrieve_for_section returns merged chunks with chunk_id, page_number, doc_id, chunk_text, section."""
    fake_hits = [
        {"chunk_id": "c_2_0", "page_number": 2, "doc_id": "doc1", "chunk_text": "Deductible is $500.", "distance": 0.1},
        {"chunk_id": "c_1_0", "page_number": 1, "doc_id": "doc1", "chunk_text": "Copay $25.", "distance": 0.2},
    ]
    with patch("backend.storage.query", return_value=fake_hits):
        results = retrieve_for_section("doc1", "Cost Summary", top_k_per_query=5, max_chunks=10)
    assert len(results) >= 1
    for r in results:
        assert "chunk_id" in r
        assert "page_number" in r
        assert "doc_id" in r
        assert "chunk_text" in r
        assert r.get("section") == "Cost Summary"
    # Sorted by page then chunk_id
    assert results == sorted(results, key=lambda x: (x["page_number"], x["chunk_id"]))


def test_retrieve_for_section_dedupes_and_caps() -> None:
    """Same chunk from two queries appears once; total count capped by max_chunks."""
    hit = {"chunk_id": "c_1_0", "page_number": 1, "doc_id": "d", "chunk_text": "Same.", "distance": 0.0}
    with patch("backend.storage.query", return_value=[hit]):
        results = retrieve_for_section("d", "Plan Snapshot", top_k_per_query=2, max_chunks=5)
    # Same chunk returned for each of the 3 Plan Snapshot queries, but deduped to one
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c_1_0"
