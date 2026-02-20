"""Tests for vector store (Chroma). Uses fake embedding to avoid API key."""

from pathlib import Path
from unittest.mock import patch

import pytest

from backend.schemas import Chunk
from backend.storage import add_chunks, query

# text-embedding-3-small dimension
EMBEDDING_DIM = 1536


class FakeEmbeddingFunction:
    """Returns fixed-dim vectors so Chroma works without OpenAI."""

    def __call__(self, input_texts: list[str]):
        return [[0.0] * EMBEDDING_DIM for _ in input_texts]


@pytest.fixture
def temp_chroma_path(tmp_path: Path):
    return tmp_path / "chroma_test"


def test_query_empty_text_returns_empty_list() -> None:
    assert query("any-doc", "   ") == []
    assert query("any-doc", "") == []


def test_add_chunks_empty_list_no_op() -> None:
    """add_chunks with empty list should not raise."""
    with patch("backend.storage._get_collection"):
        add_chunks("doc1", [])


def test_add_chunks_and_query_structure(
    temp_chroma_path: Path,
) -> None:
    """With mocked client path and fake embeddings, add then query returns expected keys."""
    mock_settings = type("S", (), {})()
    mock_settings.get_vector_db_path_resolved = lambda: temp_chroma_path
    mock_settings.openai_api_key = "test-key"
    mock_settings.embedding_model = "text-embedding-3-small"

    chunks = [
        Chunk(chunk_id="c_1_0", page_number=1, doc_id="doc1", chunk_text="Deductibles are $500."),
        Chunk(chunk_id="c_1_1", page_number=1, doc_id="doc1", chunk_text="Copay is $25."),
    ]
    with (
        patch("backend.config.get_settings", return_value=mock_settings),
        patch(
            "backend.storage._get_embedding_function",
            return_value=FakeEmbeddingFunction(),
        ),
    ):
        add_chunks("doc1", chunks)
        results = query("doc1", "deductible", top_k=2)

    assert len(results) <= 2
    for r in results:
        assert "chunk_id" in r
        assert "page_number" in r
        assert "doc_id" in r
        assert "chunk_text" in r
        assert r["doc_id"] == "doc1"
