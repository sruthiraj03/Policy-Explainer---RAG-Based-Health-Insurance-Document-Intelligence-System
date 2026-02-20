"""Tests for PDF ingestion parser."""

import tempfile
from pathlib import Path

import pytest

from backend.ingestion import extract_pages


def test_extract_pages_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        extract_pages(Path("/nonexistent/file.pdf"))


def test_extract_pages_not_pdf() -> None:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not a pdf")
        path = Path(f.name)
    try:
        with pytest.raises(ValueError, match="Not a PDF"):
            extract_pages(path)
    finally:
        path.unlink(missing_ok=True)


def test_extract_pages_empty_pdf() -> None:
    """Minimal valid PDF with zero pages (or we use a 1-page minimal PDF)."""
    import fitz
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = Path(f.name)
    try:
        doc = fitz.open()
        doc.save(path)
        doc.close()
        pages = extract_pages(path)
        assert pages == []
    finally:
        path.unlink(missing_ok=True)
