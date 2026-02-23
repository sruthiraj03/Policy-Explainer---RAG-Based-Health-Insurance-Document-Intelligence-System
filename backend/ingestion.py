"""
Ingestion Pipeline: PDF to Vector Search.
This module handles the 'ETL' (Extract, Transform, Load) for your policy documents.
It focuses on keeping citations accurate by chunking text page-by-page.
"""

import re
import shutil
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF: High-performance PDF parsing

from backend.schemas import Chunk, ExtractedPage
from backend import storage

# --- Chunking Constants ---
# These values are tuned for insurance documents, which are dense.
TARGET_TOKENS = 600  # Ideal size for the LLM to process in one 'thought'
MIN_TOKENS = 500
MAX_TOKENS = 800  # Upper limit to stay within the LLM's context window effectively
OVERLAP_TOKENS = 80  # Sentences shared between chunks so context isn't lost at the 'cut'
MIN_CHUNK_CHARS = 50  # Ignore tiny fragments that lack meaningful info


def _approx_tokens(text: str) -> int:
    """A fast, rough estimate of token count (4 chars ≈ 1 token)."""
    return max(0, len(text.strip()) // 4)


def _split_into_sentences(text: str) -> list[str]:
    """Uses regex to split text while keeping sentence endings (. ! ?) intact."""
    if not text.strip():
        return []
    # Looks for punctuation followed by space
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_page_text(page_number: int, doc_id: str, text: str) -> list[Chunk]:
    """
    Core Logic: Breaks a single page into manageable chunks.
    It preserves citations by ensuring every chunk knows its parent page_number.
    """
    if not text.strip():
        return [Chunk(chunk_id=f"c_{page_number}_0", page_number=page_number, doc_id=doc_id, chunk_text="")]

    sentences = _split_into_sentences(text)
    if not sentences:
        return [Chunk(chunk_id=f"c_{page_number}_0", page_number=page_number, doc_id=doc_id, chunk_text=text.strip())]

    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    chunk_index = 0

    for sent in sentences:
        sent_tokens = _approx_tokens(sent)

        # If adding the next sentence exceeds MAX_TOKENS, 'close' the current chunk
        if current and current_tokens + sent_tokens > MAX_TOKENS:
            chunk_text = " ".join(current)
            chunks.append(Chunk(
                chunk_id=f"c_{page_number}_{chunk_index}",
                page_number=page_number,
                doc_id=doc_id,
                chunk_text=chunk_text
            ))
            chunk_index += 1

            # --- Sliding Window Overlap ---
            # Take the end of the last chunk and put it at the start of the next one
            # so phrases aren't cut in half.
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current):
                overlap_tokens += _approx_tokens(s)
                overlap_sentences.insert(0, s)
                if overlap_tokens >= OVERLAP_TOKENS:
                    break
            current = overlap_sentences.copy()
            current_tokens = overlap_tokens

        current.append(sent)
        current_tokens += sent_tokens

    # Capture the final remaining sentences
    if current:
        chunk_text = " ".join(current)
        chunks.append(Chunk(chunk_id=f"c_{page_number}_{chunk_index}", page_number=page_number, doc_id=doc_id,
                            chunk_text=chunk_text))

    return chunks


def chunk_pages(pages: list[ExtractedPage], doc_id: str) -> list[Chunk]:
    """Helper to process a list of pages into a flat list of chunks."""
    all_chunks: list[Chunk] = []
    for page in pages:
        all_chunks.extend(_chunk_page_text(page.page_number, doc_id, page.text))
    return all_chunks


# --- PDF Parsing & Cleaning ---

def _normalize_line(s: str) -> str:
    """Removes extra whitespace and standardizes formatting."""
    return " ".join(s.split()).strip()


def _looks_like_page_number(line: str) -> bool:
    """Detects page markers like '1', 'Page 1 of 10', or '5/12' to avoid indexing them."""
    line = _normalize_line(line)
    if not line: return True
    if re.match(r"^\d+$", line): return True
    if re.match(r"^page\s+\d+\s+of\s+\d+$", line, re.I): return True
    if re.match(r"^\d+\s*/\s*\d+$", line): return True
    return False


def _clean_page_text(raw: str, drop_first_last_lines: bool = True) -> str:
    """Removes 'junk' lines from the top and bottom of pages (headers/footers)."""
    lines = [ln.strip() for ln in raw.splitlines()]
    if not lines: return ""

    if drop_first_last_lines:
        # Pop lines from top if they are short or look like page numbers
        while lines and (_looks_like_page_number(lines[0]) or len(lines[0]) < 3):
            lines.pop(0)
        # Pop lines from bottom
        while lines and (_looks_like_page_number(lines[-1]) or len(lines[-1]) < 3):
            lines.pop()

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)  # Collapse multiple newlines
    return text.strip()


def _remove_repeated_header_footer(lines_by_page: list[list[str]]) -> list[list[str]]:
    """
    Statistical Cleaning: If a line appears on >50% of pages, it is
    likely a header (like 'Company XYZ Policy') and should be removed
    to prevent it from cluttering the search results.
    """
    if len(lines_by_page) < 2:
        return lines_by_page

    all_lines: Counter[str] = Counter()
    for page_lines in lines_by_page:
        for ln in page_lines:
            n = _normalize_line(ln)
            if n and len(n) > 2:
                all_lines[n] += 1

    threshold = max(2, len(lines_by_page) // 2)
    repeated = {ln for ln, count in all_lines.items() if count > threshold}

    return [[ln for ln in page_lines if _normalize_line(ln) not in repeated] for page_lines in lines_by_page]


def extract_pages(pdf_path: Path | str, clean_headers_footers: bool = True) -> list[ExtractedPage]:
    """Uses PyMuPDF to extract text while maintaining page boundaries."""
    path = Path(pdf_path)
    # Basic file validation
    if not path.exists(): raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf": raise ValueError(f"Not a PDF file: {path}")

    doc = fitz.open(path)
    if len(doc) == 0:
        doc.close()
        raise ValueError(f"PDF has no pages: {path}")

    pages: list[ExtractedPage] = []
    raw_lines_by_page: list[list[str]] = []

    try:
        for i in range(len(doc)):
            page_num = i + 1
            page = doc[i]
            # 'sort=True' reads text from top-to-bottom, which is vital for tables
            text = page.get_text("text", sort=True) or ""

            raw_lines_by_page.append([ln.strip() for ln in text.splitlines()])
            pages.append(ExtractedPage(page_number=page_num, text=text))
    finally:
        doc.close()

    if not clean_headers_footers or not pages:
        return pages

    # Run the statistical cleaning
    cleaned_lines = _remove_repeated_header_footer(raw_lines_by_page)

    return [
        ExtractedPage(
            page_number=p.page_number,
            text=_clean_page_text("\n".join(lines))
        )
        for p, lines in zip(pages, cleaned_lines, strict=True)
    ]


# --- Execution Pipeline ---

def run_ingest(pdf_content: bytes, base_path: Path | None = None) -> str:
    """
    The Orchestrator:
    1. Generates a unique ID.
    2. Saves physical PDF.
    3. Extracts text.
    4. Creates Chunks.
    5. Stores chunks in Vector DB (add_chunks).
    """
    doc_id = storage.generate_document_id()
    pdf_path = storage.save_raw_pdf(pdf_content, doc_id, base_path)

    try:
        pages = extract_pages(pdf_path, clean_headers_footers=True)
        if not pages or sum(len(p.text or "") for p in pages) == 0:
            raise ValueError("PDF has no extractable text")

        storage.save_extracted_pages(pages, doc_id, base_path)

        chunks = chunk_pages(pages, doc_id)
        if not chunks:
            raise ValueError("PDF produced no chunks")

        storage.save_chunks(chunks, doc_id, base_path)
        storage.add_chunks(doc_id, chunks)  # This puts it in the Search Index (e.g. Chroma)

    except Exception as e:
        # Cleanup on failure to prevent orphaned files
        doc_dir = storage.get_document_dir(doc_id, base_path)
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
        raise ValueError(f"PDF processing failed: {e}") from e

    return doc_id