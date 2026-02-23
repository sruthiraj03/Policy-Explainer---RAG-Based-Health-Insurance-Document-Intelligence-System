"""
Document storage (doc_store) and vector store (Chroma).
Updated to operate in a strict "Stateless" mode:
The database is completely wiped before any new document is ingested,
ensuring zero cross-document hallucinations.
"""

import json
import uuid
from pathlib import Path
from typing import Any
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

from backend.config import get_settings
from backend.schemas import Chunk, ExtractedPage, PolicySummaryOutput
from backend.utils import cache_get, cache_invalidate, cache_set

# --- File Paths & Constants ---
DEFAULT_DOC_STORAGE_PATH = Path(__file__).resolve().parent.parent / "data" / "documents"
RAW_PDF_FILENAME = "raw.pdf"
PAGES_JSON_FILENAME = "pages.json"
CHUNKS_JSONL_FILENAME = "chunks.jsonl"
POLICY_SUMMARY_FILENAME = "Policy_summary.json"

COLLECTION_NAME = "policy_chunks"


# --- 1. Local File System Storage ---
# These functions manage the physical PDF and JSON files on your hard drive.
# Because they are saved in UUID-named folders, they naturally isolate themselves.

def generate_document_id() -> str:
    """Generates a unique ID for the current session's document."""
    return str(uuid.uuid4())


def _doc_dir(document_id: str, base_path: Path | None) -> Path:
    """Safely creates and returns the directory path for a specific document."""
    base = base_path if base_path is not None else DEFAULT_DOC_STORAGE_PATH
    base = base.resolve()
    if "/" in document_id or "\\" in document_id or document_id in (".", ".."):
        raise ValueError("Invalid document_id: must not contain path separators")
    doc_dir = base / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    return doc_dir


def get_document_dir(document_id: str, base_path: Path | None = None) -> Path:
    """Returns the storage directory path without creating it."""
    base = base_path if base_path is not None else DEFAULT_DOC_STORAGE_PATH
    base = base.resolve()
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
    """Loads the physical chunk data from the JSONL file."""
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


def get_policy_summary_path(document_id: str, base_path: Path | None = None) -> Path:
    return get_document_dir(document_id, base_path) / POLICY_SUMMARY_FILENAME


# --- 2. Vector Database Storage (Chroma) ---
# These functions manage the AI's searchable memory.

@lru_cache(maxsize=1)
def _get_client():
    """Establishes the persistent connection to the local ChromaDB folder."""
    settings = get_settings()
    path = settings.get_vector_db_path_resolved()
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _get_embedding_function():
    """Connects to OpenAI to convert text into searchable vectors."""
    settings = get_settings()
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )


def _get_collection():
    """
    Retrieves the active database collection.
    Note: @lru_cache is deliberately removed here so we always fetch a fresh
    reference in case the database was just wiped.
    """
    client = _get_client()
    ef = _get_embedding_function()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "Policy document chunks"},
    )


def wipe_database() -> None:
    """
    THE NUCLEAR OPTION: Completely destroys the vector collection.
    This guarantees that absolutely zero data from old documents survives.
    """
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🧹 DEBUG: Database completely wiped. Slate is clean.")
    except Exception:
        # If the collection doesn't exist yet, it will throw an exception. We can safely ignore it.
        pass


def add_chunks(doc_id: str, chunks: list[Chunk]) -> None:
    """Wipes the old data, then adds the new document's chunks into the database."""
    if not chunks:
        return

    # 1. Annihilate all previous vector data before doing anything else
    wipe_database()

    # 2. Get a brand new, empty collection
    collection = _get_collection()
    doc_id_str = str(doc_id)

    # 3. Format the new data
    ids = [str(c.chunk_id) for c in chunks]
    documents = [c.chunk_text for c in chunks]
    metadatas = [
        {
            "chunk_id": str(c.chunk_id),
            "page_number": int(c.page_number),
            "doc_id": doc_id_str  # Still tagging it for good measure
        } for c in chunks
    ]

    # 4. Save the new document into the fresh database
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    # 5. Force Windows to synchronize the file write
    _get_client().heartbeat()
    print(f"✅ DEBUG: New document ingested. Total DB Count: {collection.count()}")


def query(doc_id: str, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Searches the currently active database for answers."""
    if not query_text.strip():
        return []

    collection = _get_collection()
    doc_id_str = str(doc_id)

    # We still use the where_filter as an extra layer of safety
    where_filter = {"doc_id": doc_id_str}

    _get_client().heartbeat()
    print(f"🔎 DEBUG: Searching Vector DB for: '{query_text}'")

    try:
        results = collection.query(
            query_texts=[query_text.strip()],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
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