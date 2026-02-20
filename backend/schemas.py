"""API and domain models (Pydantic)."""

from typing import Literal

from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    """Single page from PDF extraction; used for citations."""

    page_number: int = Field(..., ge=1, description="1-based page number")
    text: str = Field(default="", description="Extracted text for this page")


class Chunk(BaseModel):
    """One chunk of document text for retrieval; citation-friendly (single page)."""

    chunk_id: str = Field(..., description="ID like c_{page}_{index}")
    page_number: int = Field(..., ge=1, description="1-based page number for citations")
    doc_id: str = Field(..., description="Parent document ID")
    chunk_text: str = Field(default="", description="Chunk content")


DetailLevel = Literal["standard", "detailed"]
NOT_FOUND_MESSAGE = "Not found in this document."


class Citation(BaseModel):
    """Internal citation: page + chunk_id. UI displays page only."""

    page: int = Field(..., ge=1, description="1-based page number")
    chunk_id: str = Field(..., description="Chunk id e.g. c_12_3")


class BulletWithCitations(BaseModel):
    """One summary bullet with citations."""

    text: str = Field(..., description="Plain English bullet text")
    citations: list[Citation] = Field(default_factory=list, description="page + chunk_id for each source")


class SectionSummaryOutput(BaseModel):
    """Structured output for one section: bullets with citations or not found."""

    section_name: str = Field(..., description="Section name e.g. Cost Summary")
    present: bool = Field(..., description="False when information not in document")
    bullets: list[BulletWithCitations] = Field(default_factory=list, description="4-6 (standard) or 8-12 (detailed) bullets")
    not_found_message: str | None = Field(default=None, description="Set to NOT_FOUND_MESSAGE when present is False")


ConfidenceLevel = Literal["high", "medium", "low"]

DEFAULT_DISCLAIMER = (
    "This summary is for informational purposes only. "
    "It does not replace the full policy document. "
    "See the complete policy for binding terms and conditions."
)


class DocMetadata(BaseModel):
    """Metadata extracted for the summarized document."""

    doc_id: str = Field(..., description="Document ID")
    generated_at: str = Field(..., description="ISO 8601 timestamp when summary was generated")
    total_pages: int = Field(..., ge=0, description="Number of pages in the document")
    source_file: str | None = Field(default=None, description="Original filename if known")


class SectionSummaryWithConfidence(BaseModel):
    """Section summary plus confidence label for full summary."""

    section_name: str = Field(...)
    present: bool = Field(...)
    bullets: list[BulletWithCitations] = Field(default_factory=list)
    not_found_message: str | None = Field(default=None)
    confidence: ConfidenceLevel = Field(...)
    validation_issues: list[str] = Field(default_factory=list)


class PolicySummaryOutput(BaseModel):
    """Full policy summary JSON: metadata, disclaimer, sections with confidence."""

    metadata: DocMetadata = Field(...)
    disclaimer: str = Field(...)
    sections: list[SectionSummaryWithConfidence] = Field(default_factory=list)


QA_ANSWER_TYPE = Literal["answerable", "not_found", "ambiguous", "conflict", "section_detail", "scenario"]

QA_RESPONSE_DISCLAIMER = (
    "This explanation is for informational purposes only and is not medical or legal advice. "
    "Please refer to your official policy documents or contact your insurer for confirmation."
)


class PageCitation(BaseModel):
    """Page-only citation for UI (section_detail response)."""

    page: int = Field(..., ge=1)


class QAResponseOutput(BaseModel):
    """Structured Q&A response: answer, type, citations, confidence, disclaimer."""

    doc_id: str = Field(...)
    question: str = Field(...)
    answer: str = Field(...)
    answer_type: QA_ANSWER_TYPE = Field(...)
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceLevel = Field(...)
    disclaimer: str = Field(...)
    validation_issues: list[str] = Field(default_factory=list)
    answer_display: str | None = Field(default=None)


class SectionDetailQAResponseOutput(BaseModel):
    """Q&A response for section deep-dive: bullets (8–12), page citations."""

    doc_id: str = Field(...)
    question: str = Field(...)
    answer_type: Literal["section_detail"] = Field(default="section_detail")
    section_id: str = Field(...)
    answer: str = Field(...)
    bullets: list[BulletWithCitations] = Field(default_factory=list)
    citations: list[PageCitation] = Field(default_factory=list)
    confidence: ConfidenceLevel = Field(...)
    disclaimer: str = Field(...)


class ScenarioStepOutput(BaseModel):
    """One step in an example scenario; every numeric value must be cited."""

    step_number: int = Field(..., ge=1)
    text: str = Field(...)
    citations: list[PageCitation] = Field(default_factory=list)


class ScenarioQAResponseOutput(BaseModel):
    """Example scenario response: 3–6 steps, policy-derived numbers only."""

    doc_id: str = Field(...)
    question: str = Field(...)
    answer_type: Literal["scenario"] = Field(default="scenario")
    scenario_type: str = Field(...)
    header: str = Field(...)
    steps: list[ScenarioStepOutput] = Field(default_factory=list, max_length=6)
    not_found_message: str | None = Field(default=None)
    citations: list[PageCitation] = Field(default_factory=list)
    confidence: ConfidenceLevel = Field(...)
    disclaimer: str = Field(...)


class PreventiveQAResponseOutput(QAResponseOutput):
    """Q&A response for preventive care: may include general guidance and follow-up questions."""

    follow_up_questions: list[str] = Field(default_factory=list)
