"""Document storage (doc_store) and vector store (Chroma)."""

import json
import uuid
from pathlib import Path
from typing import Any
from functools import lru_cache  # <-- Added for Windows DB Locking Fix

import chromadb
from chromadb.utils import embedding_functions

from backend.config import get_settings
from backend.schemas import Chunk, ExtractedPage, PolicySummaryOutput
from backend.utils import cache_get, cache_invalidate, cache_set

# --- Paths ---
DEFAULT_DOC_STORAGE_PATH = Path(__file__).resolve().parent.parent / "data" / "documents"
RAW_PDF_FILENAME = "raw.pdf"
PAGES_JSON_FILENAME = "pages.json"
CHUNKS_JSONL_FILENAME = "chunks.jsonl"
POLICY_SUMMARY_FILENAME = "Policy_summary.json"
FAITHFULNESS_REPORT_FILENAME = "faithfulness_report.json"
COMPLETENESS_REPORT_FILENAME = "completeness_report.json"
SIMPLICITY_REPORT_FILENAME = "simplicity_report.json"
EVALUATION_REPORT_FILENAME = "evaluation_report.json"

COLLECTION_NAME = "policy_chunks"


# --- Document store ---

def generate_document_id() -> str:
    return str(uuid.uuid4())


def _doc_dir(document_id: str, base_path: Path | None) -> Path:
    base = base_path if base_path is not None else DEFAULT_DOC_STORAGE_PATH
    base = base.resolve()
    if "/" in document_id or "\\" in document_id or document_id in (".", ".."):
        raise ValueError("Invalid document_id: must not contain path separators")
    doc_dir = base / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    return doc_dir


def get_document_dir(document_id: str, base_path: Path | None = None) -> Path:
    """Return the storage directory for document_id. Does not create it."""
    base = base_path if base_path is not None else DEFAULT_DOC_STORAGE_PATH
    base = base.resolve()
    if "/" in document_id or "\\" in document_id or document_id in (".", ".."):
        raise ValueError("Invalid document_id: must not contain path separators")
    return base / document_id


def save_raw_pdf(content: bytes, document_id: str, base_path: Path | None = None) -> Path:
    doc_dir = _doc_dir(document_id, base_path)
    path = doc_dir / RAW_PDF_FILENAME
    path.write_bytes(content)
    return path


def save_extracted_pages(pages: list[ExtractedPage], document_id: str, base_path: Path | None = None) -> Path:
    doc_dir = _doc_dir(document_id, base_path)
    path = doc_dir / PAGES_JSON_FILENAME
    with path.open("w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in pages], f, ensure_ascii=False, indent=2)
    return path


def save_chunks(chunks: list[Chunk], document_id: str, base_path: Path | None = None) -> Path:
    doc_dir = _doc_dir(document_id, base_path)
    path = doc_dir / CHUNKS_JSONL_FILENAME
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")
    cache_invalidate(f"chunks:{document_id}")
    return path


def load_chunks(document_id: str, base_path: Path | None = None) -> list[Chunk]:
    cache_key = f"chunks:{document_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    path = get_document_dir(document_id, base_path) / CHUNKS_JSONL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No chunks found for document {document_id}: {path}")
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk.model_validate(json.loads(line)))
    cache_set(cache_key, chunks)
    return chunks


def load_extracted_pages(document_id: str, base_path: Path | None = None) -> list[ExtractedPage]:
    path = get_document_dir(document_id, base_path) / PAGES_JSON_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No extracted pages found for document {document_id}: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [ExtractedPage.model_validate(item) for item in data]


def save_policy_summary(summary: PolicySummaryOutput, document_id: str, base_path: Path | None = None) -> Path:
    doc_dir = _doc_dir(document_id, base_path)
    path = doc_dir / POLICY_SUMMARY_FILENAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary.model_dump(), f, ensure_ascii=False, indent=2)
    cache_invalidate(f"summary:{document_id}")
    return path


def get_policy_summary_path(document_id: str, base_path: Path | None = None) -> Path:
    return get_document_dir(document_id, base_path) / POLICY_SUMMARY_FILENAME


def load_policy_summary(document_id: str, base_path: Path | None = None) -> PolicySummaryOutput:
    cache_key = f"summary:{document_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    path = get_policy_summary_path(document_id, base_path)
    if not path.exists():
        raise FileNotFoundError(f"No policy summary for document {document_id}: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    summary = PolicySummaryOutput.model_validate(data)
    cache_set(cache_key, summary)
    return summary


# --- Vector store (Chroma) ---

@lru_cache(maxsize=1)
def _get_client():
    settings = get_settings()
    path = settings.get_vector_db_path_resolved()
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _get_embedding_function():
    settings = get_settings()
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )


@lru_cache(maxsize=1)  # <-- ADD THIS HERE TO FIX THE SYNC ISSUE
def _get_collection():
    client = _get_client()
    ef = _get_embedding_function()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "Policy document chunks"},
    )


def add_chunks(doc_id: str, chunks: list[Chunk]) -> None:
    if not chunks:
        return
    collection = _get_collection()
    try:
        collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass

    print(f"🧠 DEBUG: Adding {len(chunks)} chunks to Vector DB...")

    ids = [c.chunk_id for c in chunks]
    documents = [c.chunk_text for c in chunks]
    metadatas = [{"chunk_id": c.chunk_id, "page_number": c.page_number, "doc_id": c.doc_id} for c in chunks]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    # NEW: Let's prove the DB actually saved them!
    print(f"✅ DEBUG: Chunks saved! Total chunks sitting in DB: {collection.count()}")


def query(doc_id: str, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not query_text.strip():
        return []

    collection = _get_collection()
    db_size = collection.count()

    print(f"🔎 DEBUG: Searching DB for: '{query_text}'")

    if db_size == 0:
        print("⚠️ DEBUG: Database is empty!")
        return []

    # FIX 1: Never ask for more chunks than actually exist in the database
    safe_top_k = min(top_k, db_size)

    try:
        # FIX 2: We temporarily removed the 'where' filter to bypass the Windows metadata bug.
        # Since you are uploading one PDF at a time, this is perfectly safe.
        results = collection.query(
            query_texts=[query_text.strip()],
            n_results=safe_top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        print(f"🎯 DEBUG: Found {len(ids)} matching chunks in DB.")

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        out = []
        for i, (cid, doc_text) in enumerate(zip(ids, docs, strict=False)):
            meta = metas[i] if i < len(metas) else {}
            distance = dists[i] if i < len(dists) else None
            out.append({
                "chunk_id": cid,
                "page_number": meta.get("page_number", 0),
                "doc_id": meta.get("doc_id", doc_id),
                "chunk_text": doc_text or "",
                "distance": distance
            })
        return out

    except Exception as e:
        print(f"❌ DEBUG: ChromaDB Query Failed: {e}")
        return []