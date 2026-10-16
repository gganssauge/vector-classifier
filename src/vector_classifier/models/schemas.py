"""Pydantic schemas for requests, responses, anchors, and audit documents."""

import hashlib
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


def _normalize_raw_dict(doc: Any) -> dict[str, Any]:
    """Convert input document object or model to dictionary."""
    if isinstance(doc, dict):
        return doc
    if hasattr(doc, "model_dump"):
        return doc.model_dump()
    if hasattr(doc, "__dict__"):
        return doc.__dict__
    return {}


def _extract_paperless_id(doc: dict[str, Any]) -> int | str:
    """Extract paperless_id, id, or pk from raw document dict."""
    raw_id = (
        doc.get("paperless_id")
        if doc.get("paperless_id") is not None
        else doc.get("id")
        if doc.get("id") is not None
        else doc.get("pk")
    )
    if raw_id is None:
        return ""
    try:
        return int(raw_id)
    except ValueError, TypeError:
        return str(raw_id).strip()


def _extract_current_doc_type(doc: dict[str, Any]) -> str:
    """Extract legacy doc_type string from raw document dict."""
    raw = (
        doc.get("current_doc_type")
        or doc.get("doc_type_original")
        or doc.get("doc_type")
        or doc.get("document_type")
        or doc.get("document_type_name")
    )
    if isinstance(raw, dict) and "name" in raw:
        return str(raw["name"]).strip()
    return str(raw).strip() if raw is not None else ""


def _extract_tags_and_tax_status(doc: dict[str, Any]) -> tuple[list[str], bool]:
    """Extract tag strings and determine existing tax-tagged flag."""
    raw_tags = doc.get("tags") or doc.get("tags_original") or doc.get("current_tags")
    tags: list[str] = []
    if isinstance(raw_tags, list):
        for item in raw_tags:
            if isinstance(item, dict) and "name" in item:
                tags.append(str(item["name"]).strip())
            elif item is not None:
                tags.append(str(item).strip())
    elif isinstance(raw_tags, str):
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    cleaned_tags = [t for t in tags if t]

    for field in ("is_currently_tax_tagged", "is_tax_relevant"):
        val = doc.get(field)
        if isinstance(val, bool):
            return cleaned_tags, val

    tax_keywords = (
        "steuer",
        "tax",
        "steuerrelevant",
        "steuer-review",
        "steuererklärung",
        "steuerberater",
        "werbungskosten",
        "sonderausgaben",
        "handwerker",
        "anlage v",
    )
    is_tax = any(
        any(kw in tag.strip().lower() for kw in tax_keywords) for tag in cleaned_tags
    )
    return cleaned_tags, is_tax


class AuditDocument(BaseModel):
    """Normalized document representation for batch indexing and audit evaluations."""

    paperless_id: int | str = ""
    filename: str = ""
    content: str = ""
    content_sha256: str = ""
    current_doc_type: str = ""
    current_tags: list[str] = Field(default_factory=list)
    is_currently_tax_tagged: bool = False

    @classmethod
    def from_raw(cls, doc: Any) -> "AuditDocument":
        """Normalize raw dictionary or document object into a typed AuditDocument."""
        if isinstance(doc, AuditDocument):
            return doc

        raw_dict = _normalize_raw_dict(doc)
        if not raw_dict:
            return cls()

        content = str(
            raw_dict.get("content")
            or raw_dict.get("body")
            or raw_dict.get("text")
            or ""
        )
        filename = str(
            raw_dict.get("original_file_name")
            or raw_dict.get("filename")
            or raw_dict.get("title")
            or ""
        )
        sha256 = str(
            raw_dict.get("content_sha256")
            or raw_dict.get("sha256")
            or raw_dict.get("hash")
            or (hashlib.sha256(content.encode("utf-8")).hexdigest() if content else "")
        )
        tags, is_tax = _extract_tags_and_tax_status(raw_dict)

        return cls(
            paperless_id=_extract_paperless_id(raw_dict),
            filename=filename,
            content=content,
            content_sha256=sha256,
            current_doc_type=_extract_current_doc_type(raw_dict),
            current_tags=tags,
            is_currently_tax_tagged=is_tax,
        )
