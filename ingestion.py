"""Ingestion: PDF parsing, chunking, pipeline (save → extract → chunk → index)."""

import re
import shutil
from collections import Counter
from pathlib import Path

import fitz

from backend.schemas import Chunk, ExtractedPage
from backend import storage


# --- Chunking constants ---
TARGET_TOKENS = 600
MIN_TOKENS = 500
MAX_TOKENS = 800
OVERLAP_TOKENS = 80
MIN_CHUNK_CHARS = 50


def _approx_tokens(text: str) -> int:
    return max(0, len(text.strip()) // 4)


def _split_into_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_page_text(page_number: int, doc_id: str, text: str) -> list[Chunk]:
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
        if current and current_tokens + sent_tokens > MAX_TOKENS:
            chunk_text = " ".join(current)
            chunks.append(Chunk(chunk_id=f"c_{page_number}_{chunk_index}", page_number=page_number, doc_id=doc_id, chunk_text=chunk_text))
            chunk_index += 1
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
    if current:
        chunk_text = " ".join(current)
        chunks.append(Chunk(chunk_id=f"c_{page_number}_{chunk_index}", page_number=page_number, doc_id=doc_id, chunk_text=chunk_text))
    return chunks


def chunk_pages(pages: list[ExtractedPage], doc_id: str) -> list[Chunk]:
    """Chunk extracted pages by page first, ~500–800 tokens per chunk with overlap."""
    all_chunks: list[Chunk] = []
    for page in pages:
        all_chunks.extend(_chunk_page_text(page.page_number, doc_id, page.text))
    return all_chunks


# --- PDF parsing ---

def _normalize_line(s: str) -> str:
    return " ".join(s.split()).strip()


def _looks_like_page_number(line: str) -> bool:
    line = _normalize_line(line)
    if not line:
        return True
    if re.match(r"^\d+$", line):
        return True
    if re.match(r"^page\s+\d+\s+of\s+\d+$", line, re.I):
        return True
    if re.match(r"^\d+\s*/\s*\d+$", line):
        return True
    return False


def _clean_page_text(raw: str, drop_first_last_lines: bool = True) -> str:
    lines = [ln.strip() for ln in raw.splitlines()]
    if not lines:
        return ""
    if drop_first_last_lines:
        while lines and (_looks_like_page_number(lines[0]) or len(lines[0]) < 3):
            lines.pop(0)
            if not lines:
                return ""
        while lines and (_looks_like_page_number(lines[-1]) or len(lines[-1]) < 3):
            lines.pop()
            if not lines:
                return ""
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_repeated_header_footer(lines_by_page: list[list[str]]) -> list[list[str]]:
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
    """Extract text from a PDF page-by-page. Returns list of ExtractedPage (1-based page numbers)."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise RuntimeError(f"Could not open PDF {path}: {e}") from e
    if len(doc) == 0:
        doc.close()
        raise ValueError(f"PDF has no pages: {path}")
    pages: list[ExtractedPage] = []
    raw_lines_by_page: list[list[str]] = []
    try:
        for i in range(len(doc)):
            page_num = i + 1
            text = ""
            try:
                page = doc[i]
                text = page.get_text("text", sort=True)
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""
            except Exception:
                text = ""
            raw_lines_by_page.append([ln.strip() for ln in text.splitlines()])
            pages.append(ExtractedPage(page_number=page_num, text=text))
    finally:
        doc.close()
    if not clean_headers_footers or not pages:
        return pages
    cleaned_lines = _remove_repeated_header_footer(raw_lines_by_page)
    return [ExtractedPage(page_number=p.page_number, text=_clean_page_text("\n".join(lines), drop_first_last_lines=True)) for p, lines in zip(pages, cleaned_lines, strict=True)]


# --- Pipeline ---

def run_ingest(pdf_content: bytes, base_path: Path | None = None) -> str:
    """Ingest a policy PDF: save, extract pages, chunk, save chunks and pages, index. Returns doc_id."""
    doc_id = storage.generate_document_id()
    pdf_path = storage.save_raw_pdf(pdf_content, doc_id, base_path)
    try:
        pages = extract_pages(pdf_path, clean_headers_footers=True)
        if not pages:
            raise ValueError("PDF has no pages")
        if sum(len(p.text or "") for p in pages) == 0:
            raise ValueError("PDF has no extractable text")
        storage.save_extracted_pages(pages, doc_id, base_path)
        chunks = chunk_pages(pages, doc_id)
        if not chunks:
            raise ValueError("PDF produced no chunks after processing")
        storage.save_chunks(chunks, doc_id, base_path)
        storage.add_chunks(doc_id, chunks)
    except (ValueError, RuntimeError) as e:
        try:
            doc_dir = storage.get_document_dir(doc_id, base_path)
            if doc_dir.exists():
                shutil.rmtree(doc_dir)
        except Exception:
            pass
        raise ValueError(f"PDF processing failed: {e}") from e
    return doc_id
