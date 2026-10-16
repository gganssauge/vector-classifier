"""Batch-Indexer & Offline-Audit-Runner for Vector Classification.

Evaluates semantic classification and hybrid tax relevance against legacy documents,
generates detailed CSV audit reports, and provides Rich console summaries.
"""

import asyncio
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from vector_classifier.config import settings
from vector_classifier.models.schemas import AuditDocument
from vector_classifier.vectors.classifier import (
    VectorClassifierEngine,
    extract_header_content,
)
from vector_classifier.vectors.qdrant import QdrantStorageManager, generate_point_id

logger = logging.getLogger(__name__)

AUDIT_CSV_COLUMNS: list[str] = [
    "paperless_id",
    "filename",
    "content_sha256",
    "current_doc_type",
    "predicted_doc_type",
    "doc_type_confidence",
    "doc_type_match",
    "current_tags",
    "predicted_tags",
    "tag_confidence_scores",
    "is_currently_tax_tagged",
    "predicted_tax_relevant",
    "tax_confidence",
    "tax_category",
    "tax_action",
]

TAX_KEYWORDS: tuple[str, ...] = (
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


class AuditRecord(BaseModel):
    """Evaluation record for a single processed document."""

    paperless_id: int | str = ""
    filename: str = ""
    content_sha256: str = ""
    current_doc_type: str = ""
    predicted_doc_type: str = ""
    doc_type_confidence: float = 0.0
    doc_type_match: bool = False
    current_tags: str = ""
    predicted_tags: str = ""
    tag_confidence_scores: str = "{}"
    is_currently_tax_tagged: bool = False
    predicted_tax_relevant: bool = False
    tax_confidence: float = 0.0
    tax_category: str = ""
    tax_action: str = "none"

    def to_csv_dict(self) -> dict[str, Any]:
        """Convert record to dictionary matching AUDIT_CSV_COLUMNS."""
        return {
            "paperless_id": self.paperless_id if self.paperless_id is not None else "",
            "filename": self.filename,
            "content_sha256": self.content_sha256,
            "current_doc_type": self.current_doc_type,
            "predicted_doc_type": self.predicted_doc_type,
            "doc_type_confidence": self.doc_type_confidence,
            "doc_type_match": self.doc_type_match,
            "current_tags": self.current_tags,
            "predicted_tags": self.predicted_tags,
            "tag_confidence_scores": self.tag_confidence_scores,
            "is_currently_tax_tagged": self.is_currently_tax_tagged,
            "predicted_tax_relevant": self.predicted_tax_relevant,
            "tax_confidence": self.tax_confidence,
            "tax_category": self.tax_category,
            "tax_action": self.tax_action,
        }


class AuditSummary(BaseModel):
    """Aggregate statistics for an audit run."""

    total_documents: int = 0
    doc_type_evaluated: int = 0
    doc_type_matches: int = 0
    doc_type_accuracy: float = 0.0
    total_assigned_tags: int = 0
    newly_discovered_tax_docs: int = 0
    confident_count: int = 0
    review_count: int = 0
    none_count: int = 0


def determine_tax_action(tax_confidence: float) -> str:
    """Determine tax action based on three-zone confidence thresholds."""
    if tax_confidence >= settings.threshold_tax_confident:
        return "auto-tag"
    if tax_confidence >= settings.threshold_tax_review:
        return "review"
    return "none"


def extract_current_tags(doc: dict[str, Any]) -> list[str]:
    """Extract list of tag strings from document metadata."""
    raw_tags = doc.get("tags") or doc.get("tags_original") or doc.get("current_tags")
    if not raw_tags:
        return []
    if isinstance(raw_tags, list):
        result: list[str] = []
        for item in raw_tags:
            if isinstance(item, dict) and "name" in item:
                result.append(str(item["name"]).strip())
            elif item is not None:
                result.append(str(item).strip())
        return [t for t in result if t]
    if isinstance(raw_tags, str):
        return [t.strip() for t in raw_tags.split(",") if t.strip()]
    return []


def is_tax_tagged(doc: dict[str, Any], tags: list[str]) -> bool:
    """Check if document is already marked as tax relevant."""
    for field in ("is_currently_tax_tagged", "is_tax_relevant"):
        val = doc.get(field)
        if isinstance(val, bool):
            return val
    for tag in tags:
        tag_lower = tag.strip().lower()
        if any(kw in tag_lower for kw in TAX_KEYWORDS):
            return True
    return False


def _extract_documents_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Extract list of document dicts from parsed JSON payload."""
    if isinstance(payload, list):
        return [doc for doc in payload if isinstance(doc, dict)]
    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload["results"], list):
            return [doc for doc in payload["results"] if isinstance(doc, dict)]
        if "documents" in payload and isinstance(payload["documents"], list):
            return [doc for doc in payload["documents"] if isinstance(doc, dict)]
        if "data" in payload and isinstance(payload["data"], list):
            return [doc for doc in payload["data"] if isinstance(doc, dict)]
        return [payload]
    return []


def _parse_json_or_jsonl(text: str) -> list[dict[str, Any]]:
    """Parse text content as either JSON structure or JSONL lines."""
    text_stripped = text.strip()
    if not text_stripped:
        return []
    try:
        parsed = json.loads(text_stripped)
        return _extract_documents_from_payload(parsed)
    except json.JSONDecodeError:
        pass

    documents: list[dict[str, Any]] = []
    for line in text_stripped.splitlines():
        line_str = line.strip()
        if line_str:
            item = json.loads(line_str)
            if isinstance(item, dict):
                documents.append(item)
    return documents


def load_documents(
    source: str | Path, limit: int | None = None
) -> list[dict[str, Any]]:
    """Load documents from JSON/JSONL file path or HTTP/HTTPS Paperless API URL."""
    source_str = str(source)
    if source_str.startswith(("http://", "https://")):
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(source_str)
            resp.raise_for_status()
            try:
                parsed = resp.json()
                documents = _extract_documents_from_payload(parsed)
            except Exception:
                documents = _parse_json_or_jsonl(resp.text)
    else:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Input file not found: {source_path}")
        text = source_path.read_text(encoding="utf-8")
        documents = _parse_json_or_jsonl(text)

    if limit is not None and limit > 0:
        documents = documents[:limit]
    return documents


def load_documents_from_file(
    source: str | Path, limit: int | None = None
) -> list[dict[str, Any]]:
    """Load documents from file or URL with optional limit (alias for load_documents)."""
    return load_documents(source, limit=limit)


class AuditRunner:
    """Coordinates batch document classification, indexing, and audit reporting."""

    def __init__(
        self,
        engine: VectorClassifierEngine | None = None,
        qdrant_manager: QdrantStorageManager | None = None,
        concurrency: int = 4,
        index: bool = False,
        dry_run: bool = True,
        target_collection: str | None = None,
    ) -> None:
        self.engine = engine if engine is not None else VectorClassifierEngine()
        self.target_collection = target_collection
        if qdrant_manager is not None:
            self.qdrant_manager = qdrant_manager
            if target_collection:
                self.qdrant_manager.collection = target_collection
        elif index or not dry_run or target_collection:
            self.qdrant_manager = QdrantStorageManager(collection=target_collection)
        else:
            self.qdrant_manager = None

        self.concurrency = max(1, concurrency)
        self.index = index
        self.dry_run = dry_run
        self._owns_engine = engine is None

    async def close(self) -> None:
        """Close managed clients."""
        if self._owns_engine and self.engine is not None:
            await self.engine.close()
        if self.qdrant_manager is not None:
            await self.qdrant_manager.close()

    async def process_document(
        self, doc: AuditDocument | dict[str, Any]
    ) -> AuditRecord:
        """Classify a single document and optionally index into Qdrant."""
        audit_doc = AuditDocument.from_raw(doc)
        content = audit_doc.content
        filename = audit_doc.filename
        paperless_id = audit_doc.paperless_id
        sha256 = audit_doc.content_sha256
        current_doc_type = audit_doc.current_doc_type
        current_tags_list = audit_doc.current_tags
        currently_tax_tagged = audit_doc.is_currently_tax_tagged

        # Vector Classification
        result = await self.engine.classify(
            content=content,
            filename=filename,
            sha256=sha256,
        )

        pred_doc_type = result.predicted_doc_type or ""
        doc_type_confidence = round(float(result.doc_type_confidence), 4)
        doc_type_match = bool(
            current_doc_type
            and pred_doc_type
            and current_doc_type.strip().lower() == pred_doc_type.strip().lower()
        )

        pred_tags_str = ", ".join(sorted(result.predicted_tags))
        current_tags_str = ", ".join(sorted(current_tags_list))
        tag_scores_json = json.dumps(result.tag_scores, ensure_ascii=False)

        tax_confidence = round(float(result.tax_confidence), 4)
        tax_action = determine_tax_action(tax_confidence)
        predicted_tax_relevant = bool(result.is_tax_relevant)

        # Optional Qdrant Upsert
        if self.index and self.qdrant_manager is not None:
            if self.target_collection:
                self.qdrant_manager.collection = self.target_collection
            header_vector = result.header_vector
            content_vector = result.content_vector
            if header_vector is None or content_vector is None:
                header_text = extract_header_content(content)
                (
                    header_vector,
                    content_vector,
                ) = await self.engine.ollama_client.get_document_embeddings(
                    header_text, content
                )

            point_id = generate_point_id(sha256)
            payload = {
                "sha256": sha256,
                "filename": filename,
                "paperless_id": paperless_id,
                "source_system": "paperless",
                "title": filename,
                "doc_type_original": current_doc_type,
                "tags_original": current_tags_list,
                "predicted_doc_type": pred_doc_type,
                "predicted_tags": result.predicted_tags,
                "is_tax_relevant": predicted_tax_relevant,
                "tax_category": result.tax_category,
                "tax_confidence": tax_confidence,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            await self.qdrant_manager.upsert_document(
                point_id=point_id,
                header_vector=header_vector,
                content_vector=content_vector,
                payload=payload,
            )

        return AuditRecord(
            paperless_id=paperless_id,
            filename=filename,
            content_sha256=sha256,
            current_doc_type=current_doc_type,
            predicted_doc_type=pred_doc_type,
            doc_type_confidence=doc_type_confidence,
            doc_type_match=doc_type_match,
            current_tags=current_tags_str,
            predicted_tags=pred_tags_str,
            tag_confidence_scores=tag_scores_json,
            is_currently_tax_tagged=currently_tax_tagged,
            predicted_tax_relevant=predicted_tax_relevant,
            tax_confidence=tax_confidence,
            tax_category=result.tax_category or "",
            tax_action=tax_action,
        )

    async def run(
        self,
        source: str | Path,
        output_csv: str | Path | None = None,
        limit: int | None = None,
        console: Console | None = None,
    ) -> tuple[list[AuditRecord], AuditSummary]:
        """Execute audit pass on documents from source."""
        raw_documents = load_documents(source, limit=limit)
        documents = [AuditDocument.from_raw(d) for d in raw_documents]

        if self.index and self.qdrant_manager is not None:
            if self.target_collection:
                self.qdrant_manager.collection = self.target_collection
            await self.qdrant_manager.ensure_collection()

        await self.engine.initialize_anchor_embeddings()

        semaphore = asyncio.Semaphore(self.concurrency)

        async def _worker(doc: AuditDocument) -> AuditRecord:
            async with semaphore:
                return await self.process_document(doc)

        tasks = [_worker(doc) for doc in documents]
        records: list[AuditRecord] = await asyncio.gather(*tasks)

        # Summary calculation
        doc_evaluated = sum(1 for r in records if bool(r.current_doc_type))
        doc_matches = sum(1 for r in records if r.doc_type_match)
        doc_accuracy = (
            round(doc_matches / doc_evaluated, 4) if doc_evaluated > 0 else 0.0
        )

        total_tags = sum(
            len([t for t in r.predicted_tags.split(", ") if t]) for r in records
        )
        newly_tax = sum(
            1
            for r in records
            if r.predicted_tax_relevant and not r.is_currently_tax_tagged
        )
        conf_count = sum(1 for r in records if r.tax_action == "auto-tag")
        rev_count = sum(1 for r in records if r.tax_action == "review")
        non_count = sum(1 for r in records if r.tax_action == "none")

        summary = AuditSummary(
            total_documents=len(records),
            doc_type_evaluated=doc_evaluated,
            doc_type_matches=doc_matches,
            doc_type_accuracy=doc_accuracy,
            total_assigned_tags=total_tags,
            newly_discovered_tax_docs=newly_tax,
            confident_count=conf_count,
            review_count=rev_count,
            none_count=non_count,
        )

        if output_csv is not None:
            self.write_csv(records, output_csv)

        if console is not None:
            self.print_summary(summary, console=console)

        return records, summary

    @staticmethod
    def write_csv(records: list[AuditRecord], output_path: str | Path) -> None:
        """Write records to CSV file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=AUDIT_CSV_COLUMNS)
            writer.writeheader()
            for r in records:
                writer.writerow(r.to_csv_dict())

    @staticmethod
    def print_summary(summary: AuditSummary, console: Console | None = None) -> None:
        """Print summary statistics using Rich console."""
        c = console or Console()
        table = Table(title="Offline Audit Summary", border_style="cyan")
        table.add_column("Metric", style="bold white")
        table.add_column("Value", style="bold yellow")

        table.add_row("Total documents processed", str(summary.total_documents))
        match_str = (
            f"{summary.doc_type_accuracy * 100:.1f}% ({summary.doc_type_matches}/{summary.doc_type_evaluated})"
            if summary.doc_type_evaluated > 0
            else "N/A"
        )
        table.add_row("Doc Type match rate / accuracy", match_str)
        table.add_row("Total thematic tags assigned", str(summary.total_assigned_tags))
        table.add_row(
            "Newly discovered tax documents", str(summary.newly_discovered_tax_docs)
        )
        table.add_row(
            "Tax Zone: Confident (auto-tag, ≥ 0.82)", str(summary.confident_count)
        )
        table.add_row("Tax Zone: Review (0.70 - 0.81)", str(summary.review_count))
        table.add_row("Tax Zone: None (< 0.70)", str(summary.none_count))

        c.print(table)


async def run_audit(
    source: str | Path,
    output_csv: str | Path | None = "audit_report.csv",
    limit: int | None = None,
    dry_run: bool = True,
    concurrency: int = 4,
    index: bool = False,
    engine: VectorClassifierEngine | None = None,
    qdrant_manager: QdrantStorageManager | None = None,
    target_collection: str | None = None,
    console: Console | None = None,
) -> tuple[list[AuditRecord], AuditSummary]:
    """Helper function to execute an audit run."""
    runner = AuditRunner(
        engine=engine,
        qdrant_manager=qdrant_manager,
        concurrency=concurrency,
        index=index,
        dry_run=dry_run,
        target_collection=target_collection,
    )
    try:
        return await runner.run(
            source=source,
            output_csv=output_csv,
            limit=limit,
            console=console,
        )
    finally:
        await runner.close()
