"""Tests for document storage."""

import json
from pathlib import Path

import pytest

from backend.schemas import ExtractedPage
from backend.storage import (
    generate_document_id,
    load_chunks,
    load_extracted_pages,
    save_chunks,
    save_extracted_pages,
    save_raw_pdf,
)


def test_generate_document_id() -> None:
    id1 = generate_document_id()
    id2 = generate_document_id()
    assert len(id1) == 36
    assert id1 != id2
    assert id1.count("-") == 4


def test_save_and_load_extracted_pages(tmp_path: Path) -> None:
    doc_id = generate_document_id()
    pages = [
        ExtractedPage(page_number=1, text="Page one"),
        ExtractedPage(page_number=2, text="Page two"),
    ]
    saved = save_extracted_pages(pages, doc_id, base_path=tmp_path)
    assert saved.exists()
    assert saved.name == "pages.json"
    loaded = load_extracted_pages(doc_id, base_path=tmp_path)
    assert loaded == pages


def test_save_raw_pdf(tmp_path: Path) -> None:
    doc_id = generate_document_id()
    content = b"%PDF-1.4 fake content"
    path = save_raw_pdf(content, doc_id, base_path=tmp_path)
    assert path.exists()
    assert path.read_bytes() == content
    assert path.name == "raw.pdf"


def test_load_extracted_pages_missing_raises(tmp_path: Path) -> None:
    doc_id = generate_document_id()
    (tmp_path / doc_id).mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No extracted pages"):
        load_extracted_pages(doc_id, base_path=tmp_path)


def test_load_chunks_missing_raises(tmp_path: Path) -> None:
    doc_id = generate_document_id()
    (tmp_path / doc_id).mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No chunks"):
        load_chunks(doc_id, base_path=tmp_path)


def test_invalid_document_id_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid document_id"):
        save_raw_pdf(b"x", "bad/id", base_path=Path("/tmp"))
