"""Tests for full summary orchestration."""

from pathlib import Path
from unittest.mock import patch

import pytest

from backend.schemas import DEFAULT_DISCLAIMER, DocMetadata, PolicySummaryOutput, SectionSummaryOutput
from backend.summarization import extract_doc_metadata, run_full_summary_pipeline


def test_extract_doc_metadata_no_pages(tmp_path: Path) -> None:
    """When pages.json is missing, total_pages is 0."""
    from backend.storage import generate_document_id
    doc_id = generate_document_id()
    (tmp_path / doc_id).mkdir(parents=True)
    meta = extract_doc_metadata(doc_id, base_path=tmp_path)
    assert meta.doc_id == doc_id
    assert meta.total_pages == 0
    assert meta.generated_at
    assert meta.source_file is None


def test_extract_doc_metadata_with_pages(tmp_path: Path) -> None:
    """When pages.json exists, total_pages is len(pages)."""
    import json
    from backend.schemas import ExtractedPage
    from backend.storage import generate_document_id, save_extracted_pages
    doc_id = generate_document_id()
    pages = [
        ExtractedPage(page_number=1, text="One"),
        ExtractedPage(page_number=2, text="Two"),
    ]
    save_extracted_pages(pages, doc_id, base_path=tmp_path)
    meta = extract_doc_metadata(doc_id, base_path=tmp_path)
    assert meta.total_pages == 2


def test_run_full_summary_pipeline_saves_policy_summary_json(tmp_path: Path) -> None:
    """Pipeline produces PolicySummaryOutput and saves to Policy_summary.json."""
    from backend.storage import generate_document_id
    doc_id = generate_document_id()
    (tmp_path / doc_id).mkdir(parents=True)
    # Mock retrieval and summarization to avoid LLM/vector store
    stub_section = SectionSummaryOutput(
        section_name="Plan Snapshot",
        present=True,
        bullets=[],
        not_found_message=None,
    )
    with (
        patch("backend.retrieval.retrieve_for_section", return_value=[]),
        patch("backend.summarization.summarize_section", return_value=stub_section),
    ):
        out = run_full_summary_pipeline(doc_id, base_path=tmp_path)
    assert isinstance(out, PolicySummaryOutput)
    assert out.metadata.doc_id == doc_id
    assert out.disclaimer == DEFAULT_DISCLAIMER
    assert len(out.sections) == 6  # all CORE_SECTIONS
    path = tmp_path / doc_id / "Policy_summary.json"
    assert path.exists()
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"]["doc_id"] == doc_id
    assert "disclaimer" in data
    assert len(data["sections"]) == 6
