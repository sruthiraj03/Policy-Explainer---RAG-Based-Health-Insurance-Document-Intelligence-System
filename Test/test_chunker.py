"""Tests for chunking strategy."""

from pathlib import Path

import pytest

from backend.ingestion import chunk_pages
from backend.schemas import Chunk, ExtractedPage
from backend.storage import load_chunks, save_chunks


def test_chunk_id_format() -> None:
    """Chunk IDs are c_{page}_{index}."""
    # One short page -> one chunk
    pages = [ExtractedPage(page_number=1, text="Short.")]
    chunks = chunk_pages(pages, doc_id="test-doc")
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c_1_0"
    assert chunks[0].page_number == 1
    assert chunks[0].doc_id == "test-doc"


def test_chunk_by_page_first() -> None:
    """Chunks from different pages have correct page_number and doc_id."""
    pages = [
        ExtractedPage(page_number=1, text="Page one content."),
        ExtractedPage(page_number=2, text="Page two content."),
    ]
    chunks = chunk_pages(pages, doc_id="doc-1")
    assert all(c.doc_id == "doc-1" for c in chunks)
    assert [c.page_number for c in chunks] == [1, 2]


def test_empty_page_produces_one_chunk() -> None:
    """Empty page still gets one chunk (empty text) for consistent structure."""
    pages = [ExtractedPage(page_number=1, text="")]
    chunks = chunk_pages(pages, doc_id="doc")
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c_1_0"
    assert chunks[0].chunk_text == ""


def test_save_and_load_chunks_jsonl(tmp_path: Path) -> None:
    """Chunks save to chunks.jsonl and load back correctly."""
    from backend.storage import generate_document_id

    doc_id = generate_document_id()
    chunks = [
        Chunk(chunk_id="c_1_0", page_number=1, doc_id=doc_id, chunk_text="First."),
        Chunk(chunk_id="c_1_1", page_number=1, doc_id=doc_id, chunk_text="Second."),
    ]
    save_chunks(chunks, doc_id, base_path=tmp_path)
    loaded = load_chunks(doc_id, base_path=tmp_path)
    assert loaded == chunks
