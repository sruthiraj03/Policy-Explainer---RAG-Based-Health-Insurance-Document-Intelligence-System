"""
API and Domain Models (Pydantic).
This module defines the 'contracts' between your LLM, your backend, and your UI.
Using Pydantic ensures that if the LLM hallucinates a field name, the code fails
early (Validation Error) rather than passing bad data to the user.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

# --- Constants & Literals ---

# define a strict schema for the LLM to follow (trying to limit the issues with hallucination)
SectionName = Literal[
    "Plan Snapshot",
    "Cost Summary",
    "Summary of Covered Services",
    "Administrative Conditions",
    "Exclusions & Limitations",
    "Claims, Appeals & Member Rights"
]

# Standardizing the confidence level (instead of numeric)
ConfidenceLevel = Literal["high", "medium", "low"]

# Categorizes the type of Q&A response for different UI rendering logic.
QA_ANSWER_TYPE = Literal[
    "answerable",    # Direct answer found in doc
    "not_found",     # Information missing; refer to website
    "ambiguous",     # Doc is unclear
    "conflict",      # Two parts of the doc say different things
    "section_detail",# Deep dive into one policy section
    "scenario"       # Example: 'If I break my leg, what do I pay?'
]

NOT_FOUND_MESSAGE = "Not found in this document."
DEFAULT_DISCLAIMER = (
    "This summary is for informational purposes only. It does not replace the full policy document."
)

# --- Base Components ---

class Citation(BaseModel):
    """
    The 'Evidence' for a claim.
    By requiring chunk_id, you can prove exactly which part of the
    Vector Database (Chroma) the LLM used.
    """
    page: int = Field(..., ge=1, description="1-based page number from the PDF")
    chunk_id: str = Field(..., description="The unique ID of the text chunk (e.g., c_1_15)")

class BulletWithCitations(BaseModel):
    """
    A single piece of information.
    Instead of a giant block of text, we force the LLM to provide 'Atomic'
    bullets where every single claim has a source citation.
    """
    text: str = Field(..., description="The policy detail written in plain English")
    citations: list[Citation] = Field(default_factory=list, description="Source links for this bullet")

# --- Section Summaries ---

class SectionSummaryBase(BaseModel):
    """
    Matches your summary_schema.json perfectly.
    'present' is the 'Truth Flag'—if the LLM can't find info for a section,
    it must mark this False instead of hallucinating.
    """
    section_name: SectionName
    present: bool
    bullets: list[BulletWithCitations] = Field(default_factory=list)
    not_found_message: Optional[str] = Field(default=None)

class SectionSummaryWithConfidence(SectionSummaryBase):
    """
    Extension of the base schema for internal evaluation.
    'validation_issues' allows you to track if the LLM missed specific
    details (like missing out-of-network costs).
    """
    confidence: ConfidenceLevel
    validation_issues: list[str] = Field(
        default_factory=list,
        description="Notes on missing data or extraction errors"
    )

# --- Full Policy Summary ---

class DocMetadata(BaseModel):
    """Tracks the 'Identity' of the document being processed."""
    doc_id: str = Field(..., description="Unique hash or ID for the policy")
    generated_at: str = Field(..., description="Timestamp for versioning")
    total_pages: int = Field(..., ge=0)
    source_file: Optional[str] = None

class PolicySummaryOutput(BaseModel):
    """
    The final object sent to the frontend.
    It combines the metadata, the legal disclaimer, and all extracted sections.
    """
    metadata: DocMetadata
    disclaimer: str = Field(default=DEFAULT_DISCLAIMER)
    sections: list[SectionSummaryWithConfidence]

# --- Q&A Components ---

class QAResponseOutput(BaseModel):
    """
    The output model for the 'Ask a Question' feature.
    Ensures that every answer the user gets is grounded and cited.
    """
    doc_id: str
    question: str
    answer: str
    answer_type: QA_ANSWER_TYPE
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceLevel
    disclaimer: str = Field(default=DEFAULT_DISCLAIMER)
    validation_issues: list[str] = Field(default_factory=list)

# -- Chunking and Extracting --
class ExtractedPage(BaseModel):
    """
    Represents the raw text of a single PDF page.
    Used during the extraction phase before chunking.
    """
    page_number: int = Field(..., ge=1, description="1-based page number")
    text: str = Field(default="", description="The full cleaned text of the page")

class Chunk(BaseModel):
    """
    A specific segment of text used for Vector Search.
    This is the 'unit of retrieval' for your RAG system.
    """
    chunk_id: str = Field(..., description="Unique ID, usually c_{page}_{index}")
    page_number: int = Field(..., ge=1)
    doc_id: str = Field(..., description="ID of the parent document")
    chunk_text: str = Field(..., description="The actual text content used for embeddings")

# -- Scenario Logic --
class ScenarioStepOutput(BaseModel):
    """One step in an example scenario; every numeric value must be cited."""
    step_number: int = Field(..., ge=1)
    text: str = Field(..., description="Plain English description of the cost step")
    citations: list[Citation] = Field(default_factory=list)

class ScenarioQAResponseOutput(BaseModel):
    """Example scenario response: 3–6 steps, policy-derived numbers only."""
    doc_id: str = Field(...)
    question: str = Field(...)
    answer_type: Literal["scenario"] = Field(default="scenario")
    scenario_type: str = Field(...)
    header: str = Field(default="Example Scenario (Hypothetical – Based on Policy Terms)")
    steps: list[ScenarioStepOutput] = Field(default_factory=list)
    not_found_message: Optional[str] = Field(default=None)
    confidence: ConfidenceLevel = Field(...)
    disclaimer: str = Field(default=DEFAULT_DISCLAIMER)
