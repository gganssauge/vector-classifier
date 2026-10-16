"""Pydantic schemas for requests, responses, and anchors."""

from typing import Any

from pydantic import BaseModel, Field


class SemanticAnchor(BaseModel):
    """Semantic anchor representation for zero-shot vector comparison."""

    name: str
    category: str
    description: str
    keywords: list[str] = Field(default_factory=list)


class ClassifyRequest(BaseModel):
    """Document classification request payload."""

    content: str
    filename: str = ""
    sha256: str = ""
    header_content: str | None = None


class ClassifyResponse(BaseModel):
    """Document classification result payload."""

    predicted_doc_type: str | None = None
    doc_type_confidence: float = 0.0
    predicted_tags: list[str] = Field(default_factory=list)
    tag_scores: dict[str, float] = Field(default_factory=dict)
    is_tax_relevant: bool = False
    tax_confidence: float = 0.0
    tax_category: str | None = None
    tax_review_required: bool = False
    matched_anchors: list[dict[str, Any]] = Field(default_factory=list)


class VectorClassificationResult(ClassifyResponse):
    """Result of vector classification pipeline."""

    header_vector: list[float] | None = None
    content_vector: list[float] | None = None


class HealthResponse(BaseModel):
    """Health check status response."""

    status: str = "ok"
    version: str = "0.1.0"
    ollama_connected: bool = False
    qdrant_connected: bool = False
