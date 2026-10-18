"""High-performance Paperless document ingestion pipeline for Qdrant vector storage."""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel
from qdrant_client import models
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from vector_classifier.config import settings
from vector_classifier.models.schemas import AuditDocument
from vector_classifier.vectors.classifier import VectorClassifierEngine
from vector_classifier.vectors.qdrant import QdrantStorageManager, generate_point_id

logger = logging.getLogger(__name__)


class IngestSummary(BaseModel):
    """Aggregate statistics for an ingestion run."""

    total_scanned: int = 0
    total_indexed: int = 0
    total_skipped_empty: int = 0
    tax_relevant_count: int = 0
    elapsed_seconds: float = 0.0
    throughput_docs_per_sec: float = 0.0


class PaperlessIngestRunner:
    """Manages streaming extraction from Paperless REST API and parallel vectorization into Qdrant."""

    def __init__(
        self,
        paperless_url: str | None = None,
        paperless_token: str | None = None,
        engine: VectorClassifierEngine | None = None,
        qdrant_manager: QdrantStorageManager | None = None,
        concurrency: int = 8,
        batch_size: int = 50,
        recreate_collection: bool = False,
    ) -> None:
        self.paperless_url = (paperless_url or settings.paperless_base_url).rstrip("/")
        self.paperless_token = paperless_token or settings.paperless_api_token
        self.concurrency = max(1, concurrency)
        self.batch_size = max(1, batch_size)
        self.recreate = recreate_collection

        self.engine = engine if engine is not None else VectorClassifierEngine()
        self.qdrant_manager = (
            qdrant_manager if qdrant_manager is not None else QdrantStorageManager()
        )
        self._owns_engine = engine is None
        self._owns_qdrant = qdrant_manager is None

        self.tags_map: dict[int, str] = {}
        self.doc_types_map: dict[int, str] = {}

    async def close(self) -> None:
        """Clean up managed engine and Qdrant connections."""
        if self._owns_engine and self.engine is not None:
            await self.engine.close()
        if self._owns_qdrant and self.qdrant_manager is not None:
            await self.qdrant_manager.close()

    async def _fetch_metadata_maps(self, client: httpx.AsyncClient) -> None:
        """Fetch tags and document types mappings from Paperless API."""
        try:
            tags_resp = await client.get(
                f"{self.paperless_url}/api/tags/?page_size=1000"
            )
            if tags_resp.status_code == 200:
                self.tags_map = {
                    t["id"]: t["name"] for t in tags_resp.json().get("results", [])
                }
        except Exception as e:
            logger.warning("Could not fetch tags mapping from Paperless: %s", e)

        try:
            dt_resp = await client.get(
                f"{self.paperless_url}/api/document_types/?page_size=1000"
            )
            if dt_resp.status_code == 200:
                self.doc_types_map = {
                    d["id"]: d["name"] for d in dt_resp.json().get("results", [])
                }
        except Exception as e:
            logger.warning(
                "Could not fetch document types mapping from Paperless: %s", e
            )

    def _resolve_document_dict(self, raw_doc: dict[str, Any]) -> dict[str, Any]:
        """Resolve numeric IDs in document dict to canonical strings."""
        doc = dict(raw_doc)
        dt_val = doc.get("document_type")
        if isinstance(dt_val, int) and dt_val in self.doc_types_map:
            doc["current_doc_type"] = self.doc_types_map[dt_val]

        tags_val = doc.get("tags")
        if isinstance(tags_val, list):
            resolved_tags: list[str] = []
            for t in tags_val:
                if isinstance(t, int) and t in self.tags_map:
                    resolved_tags.append(self.tags_map[t])
                elif isinstance(t, str):
                    resolved_tags.append(t)
            doc["tags"] = resolved_tags

        return doc

    async def _process_and_create_point(
        self,
        doc: AuditDocument,
        semaphore: asyncio.Semaphore,
    ) -> models.PointStruct | None:
        """Vectorize a single document and return a Qdrant PointStruct."""
        if not doc.content or not doc.content.strip():
            return None

        try:
            async with semaphore:
                result = await self.engine.classify(
                    content=doc.content,
                    filename=doc.filename,
                    sha256=doc.content_sha256,
                )

            header_vec = result.header_vector
            content_vec = result.content_vector
            if header_vec is None or content_vec is None:
                return None

            point_id = generate_point_id(doc.content_sha256)
            payload = {
                "sha256": doc.content_sha256,
                "filename": doc.filename,
                "paperless_id": doc.paperless_id,
                "source_system": "paperless",
                "title": doc.filename,
                "doc_type_original": doc.current_doc_type,
                "tags_original": doc.current_tags,
                "predicted_doc_type": result.predicted_doc_type or "",
                "predicted_tags": result.predicted_tags,
                "is_tax_relevant": bool(result.is_tax_relevant),
                "tax_category": result.tax_category,
                "tax_confidence": round(float(result.tax_confidence), 4),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }

            return models.PointStruct(
                id=point_id,
                vector={
                    "header": header_vec,
                    "content": content_vec,
                },
                payload=payload,
            )
        except Exception as exc:
            logger.warning(
                "Error vectorizing doc %s (%s): %s", doc.paperless_id, doc.filename, exc
            )
            return None

    async def _get_existing_point_ids(self) -> set[str]:
        """Fetch all existing point IDs from Qdrant collection for resumability."""
        existing_point_ids: set[str] = set()
        if self.recreate:
            return existing_point_ids
        try:
            next_page_offset = None
            while True:
                records, next_page_offset = await self.qdrant_manager.client.scroll(
                    collection_name=self.qdrant_manager.collection,
                    limit=1000,
                    with_payload=False,
                    with_vectors=False,
                    offset=next_page_offset,
                )
                for r in records:
                    existing_point_ids.add(str(r.id))
                if next_page_offset is None:
                    break
        except Exception as e:
            logger.warning("Could not pre-fetch existing points: %s", e)
        return existing_point_ids

    async def _process_chunk(
        self,
        chunk: list[AuditDocument],
        semaphore: asyncio.Semaphore,
        existing_point_ids: set[str],
    ) -> tuple[int, int, int]:
        """Vectorize and upsert a chunk of documents, returning (indexed, skipped, tax_count)."""
        docs_to_embed = [
            d
            for d in chunk
            if generate_point_id(d.content_sha256) not in existing_point_ids
        ]
        if not docs_to_embed:
            return 0, 0, 0

        vectorize_tasks = [
            self._process_and_create_point(d, semaphore) for d in docs_to_embed
        ]
        points = await asyncio.gather(*vectorize_tasks)
        valid_points = [p for p in points if p is not None]
        skipped = len(docs_to_embed) - len(valid_points)

        if valid_points:
            await self.qdrant_manager.upsert_documents_batch(valid_points, wait=False)
            for p in valid_points:
                existing_point_ids.add(str(p.id))
            tax_count = sum(
                1
                for p in valid_points
                if p.payload and p.payload.get("is_tax_relevant")
            )
            return len(valid_points), skipped, tax_count
        return 0, skipped, 0

    async def _process_page_chunks(
        self,
        page_docs: list[AuditDocument],
        semaphore: asyncio.Semaphore,
        existing_point_ids: set[str],
        progress: Progress,
        task: Any,
    ) -> tuple[int, int, int]:
        """Process page documents in chunks, updating progress."""
        idx_total, skp_total, tx_total = 0, 0, 0
        for i in range(0, len(page_docs), self.batch_size):
            chunk = page_docs[i : i + self.batch_size]
            idx_cnt, skp_cnt, tx_cnt = await self._process_chunk(
                chunk, semaphore, existing_point_ids
            )
            idx_total += idx_cnt
            skp_total += skp_cnt
            tx_total += tx_cnt
            progress.update(task, advance=len(chunk))
        return idx_total, skp_total, tx_total

    async def _stream_and_ingest(
        self,
        http_client: httpx.AsyncClient,
        target_count: int,
        existing_point_ids: set[str],
        progress: Progress,
        task: Any,
    ) -> tuple[int, int, int, int]:
        """Stream documents from Paperless and ingest them into Qdrant."""
        semaphore = asyncio.Semaphore(self.concurrency)
        scanned_count, indexed_count, skipped_empty, tax_relevant_count = 0, 0, 0, 0
        page = 1
        page_size = 100

        while scanned_count < target_count:
            current_page_size = min(page_size, target_count - scanned_count)
            page_url = f"{self.paperless_url}/api/documents/?page={page}&page_size={current_page_size}"
            resp = await http_client.get(page_url)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            page_docs = [
                AuditDocument.from_raw(self._resolve_document_dict(raw))
                for raw in results
            ]
            scanned_count += len(page_docs)

            idx, skp, tx = await self._process_page_chunks(
                page_docs, semaphore, existing_point_ids, progress, task
            )
            indexed_count += idx
            skipped_empty += skp
            tax_relevant_count += tx

            if not data.get("next"):
                break
            page += 1

        return scanned_count, indexed_count, skipped_empty, tax_relevant_count

    async def run(
        self,
        limit: int | None = None,
        console: Console | None = None,
    ) -> IngestSummary:
        """Execute full ingestion pipeline."""
        c = console or Console()
        start_time = time.perf_counter()

        # Step 1: Ensure Qdrant collection
        if self.recreate:
            c.print(
                f"[bold yellow]Recreating collection '{self.qdrant_manager.collection}' fresh...[/bold yellow]"
            )
            await self.qdrant_manager.recreate_collection()
        else:
            await self.qdrant_manager.ensure_collection()

        # Step 2: Initialize anchor embeddings
        c.print("[cyan]Initializing semantic anchor embeddings...[/cyan]")
        await self.engine.initialize_anchor_embeddings()

        # Step 3: Fetch metadata maps & count from Paperless
        headers = {}
        if self.paperless_token:
            headers["Authorization"] = f"Token {self.paperless_token}"

        async with httpx.AsyncClient(
            headers=headers,
            timeout=settings.paperless_timeout,
            verify=settings.paperless_ssl_verify,
        ) as http_client:
            await self._fetch_metadata_maps(http_client)

            # Probe document count
            c.print(f"[cyan]Connecting to Paperless at {self.paperless_url}...[/cyan]")
            probe_resp = await http_client.get(
                f"{self.paperless_url}/api/documents/?page_size=1"
            )
            probe_resp.raise_for_status()
            total_count = probe_resp.json().get("count", 0)

            target_count = total_count if limit is None else min(total_count, limit)
            c.print(
                f"[bold green]Found {total_count} documents in Paperless. Target: {target_count}[/bold green]"
            )

            # Step 4: Stream and ingest with progress bar
            existing_point_ids = await self._get_existing_point_ids()
            if existing_point_ids:
                c.print(
                    f"[cyan]Found {len(existing_point_ids)} already indexed points in Qdrant (will skip).[/cyan]"
                )

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=c,
            ) as progress:
                task = progress.add_task(
                    "Ingesting documents into Qdrant...", total=target_count
                )
                (
                    scanned_count,
                    indexed_count,
                    skipped_empty,
                    tax_relevant_count,
                ) = await self._stream_and_ingest(
                    http_client,
                    target_count,
                    existing_point_ids,
                    progress,
                    task,
                )

        elapsed = time.perf_counter() - start_time
        throughput = round(indexed_count / elapsed, 2) if elapsed > 0 else 0.0

        summary = IngestSummary(
            total_scanned=scanned_count,
            total_indexed=indexed_count,
            total_skipped_empty=skipped_empty,
            tax_relevant_count=tax_relevant_count,
            elapsed_seconds=round(elapsed, 2),
            throughput_docs_per_sec=throughput,
        )

        self.print_summary(summary, console=c)
        return summary

    @staticmethod
    def print_summary(summary: IngestSummary, console: Console | None = None) -> None:
        """Display summary table using Rich."""
        c = console or Console()
        table = Table(
            title="Qdrant Ingestion Summary",
            border_style="green",
        )
        table.add_column("Metric", style="bold white")
        table.add_column("Value", style="bold yellow")

        table.add_row("Total documents scanned", str(summary.total_scanned))
        table.add_row("Successfully indexed into Qdrant", str(summary.total_indexed))
        table.add_row(
            "Skipped (empty/no OCR content)", str(summary.total_skipped_empty)
        )
        table.add_row(
            "Identified tax-relevant documents", str(summary.tax_relevant_count)
        )
        table.add_row("Total elapsed time", f"{summary.elapsed_seconds:.1f} s")
        table.add_row("Throughput", f"{summary.throughput_docs_per_sec:.1f} docs/sec")

        c.print(table)
