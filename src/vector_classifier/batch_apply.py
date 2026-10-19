"""Batch runner for full-corpus groundtruth application and qualification."""

import asyncio
import logging
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

from vector_classifier.config import settings
from vector_classifier.vectors.classifier import VectorClassifierEngine
from vector_classifier.vectors.qdrant import QdrantStorageManager

logger = logging.getLogger(__name__)

CANONICAL_MACRO_TAGS = [
    "Energie",
    "Telekommunikation",
    "Fahrzeug",
    "Gesundheit",
    "Versicherung",
    "Immobilien",
    "Bank",
    "Depot",
    "Gehalt",
    "Altersvorsorge",
    "Behörde",
    "Bildung",
    "Reisen",
    "Mitgliedschaft",
]


class BatchApplyRunner:
    """Orchestrates full corpus dry-run evaluation and bulk application in Paperless."""

    def __init__(
        self,
        paperless_url: str = settings.paperless_base_url,
        paperless_token: str = settings.paperless_api_token,
        qdrant_manager: QdrantStorageManager | None = None,
        engine: VectorClassifierEngine | None = None,
    ) -> None:
        self.paperless_url = paperless_url.rstrip("/")
        self.paperless_token = paperless_token
        self.qdrant = (
            qdrant_manager if qdrant_manager is not None else QdrantStorageManager()
        )
        self.engine = (
            engine
            if engine is not None
            else VectorClassifierEngine(qdrant_manager=self.qdrant)
        )

    def _get_headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.paperless_token}"}

    async def get_or_create_tag(
        self, client: httpx.AsyncClient, name: str, color: str = "#0ea5e9"
    ) -> int:
        """Find or create a tag in Paperless."""
        resp = await client.get(f"{self.paperless_url}/api/tags/?page_size=500")
        resp.raise_for_status()
        tags = resp.json().get("results", [])
        for t in tags:
            if t["name"].strip().lower() == name.strip().lower():
                return int(t["id"])

        create_resp = await client.post(
            f"{self.paperless_url}/api/tags/",
            json={
                "name": name,
                "color": color,
                "owner": None,
                "matching_algorithm": 0,
                "match": "",
            },
        )
        create_resp.raise_for_status()
        return int(create_resp.json()["id"])

    async def _load_metadata(
        self, client: httpx.AsyncClient, dry_run: bool
    ) -> tuple[dict[str, int], dict[str, int], int, int, int, dict[str, int]]:
        """Load document types and tags from Paperless."""
        dt_resp = await client.get(
            f"{self.paperless_url}/api/document_types/?page_size=100"
        )
        dt_resp.raise_for_status()
        doc_type_map = {
            t["name"].strip(): int(t["id"]) for t in dt_resp.json().get("results", [])
        }

        tag_resp = await client.get(f"{self.paperless_url}/api/tags/?page_size=500")
        tag_resp.raise_for_status()
        tag_map = {
            t["name"].strip(): int(t["id"]) for t in tag_resp.json().get("results", [])
        }

        doctype_review_tag_id = tag_map.get(
            "Doctype-Review"
        ) or await self.get_or_create_tag(client, "Doctype-Review", color="#eab308")
        steuer_review_tag_id = (
            tag_map.get("Steuer-Review")
            or tag_map.get("steuer-review")
            or await self.get_or_create_tag(client, "Steuer-Review", color="#f59e0b")
        )
        steuerrelevant_tag_id = tag_map.get(
            "steuerrelevant"
        ) or await self.get_or_create_tag(client, "steuerrelevant", color="#10b981")

        macro_tag_ids: dict[str, int] = {}
        for mt in CANONICAL_MACRO_TAGS:
            if mt in tag_map:
                macro_tag_ids[mt] = tag_map[mt]
            elif not dry_run:
                macro_tag_ids[mt] = await self.get_or_create_tag(
                    client, mt, color="#0ea5e9"
                )

        return (
            doc_type_map,
            tag_map,
            doctype_review_tag_id,
            steuer_review_tag_id,
            steuerrelevant_tag_id,
            macro_tag_ids,
        )

    async def _evaluate_doctype(
        self,
        r: Any,
        p: dict[str, Any],
        pid: int,
        h_vec: list[float] | None,
        c_vec: list[float] | None,
        doctype_green_assign: dict[str, list[int]],
        doctype_yellow_review: list[int],
        doctype_groundtruth: list[int],
    ) -> None:
        """Evaluate document type classification for point."""
        is_dt_curated = p.get("is_doctype_curated", False)
        dt_curated = p.get("doctype_curated")
        orig_dt = p.get("doc_type_original")

        if is_dt_curated and dt_curated:
            doctype_groundtruth.append(pid)
            if dt_curated != orig_dt:
                doctype_green_assign.setdefault(dt_curated, []).append(pid)
        elif h_vec:
            _, _, matched = await self.engine._classify_doc_type(
                header_vector=h_vec,
                content_vector=c_vec,
                exclude_point_id=str(r.id),
            )
            match_info = matched[0] if matched else {}
            is_review = match_info.get("review_required", False)
            top_name = match_info.get("name")
            knn_match = match_info.get("knn_match")

            if (not is_review or knn_match) and top_name:
                chosen = knn_match or top_name
                if chosen != orig_dt:
                    doctype_green_assign.setdefault(chosen, []).append(pid)
            else:
                doctype_yellow_review.append(pid)

    async def _evaluate_tags_and_tax(
        self,
        p: dict[str, Any],
        pid: int,
        c_vec: list[float] | None,
        tags_to_add: dict[str, list[int]],
        tax_confident: list[int],
        tax_review: list[int],
    ) -> None:
        """Evaluate thematic tags and tax relevance for point."""
        if not c_vec:
            return

        pred_tags, _, _ = await self.engine._classify_thematic_tags(c_vec)
        raw_existing = p.get("tags_original") or []
        existing_tags = (
            [raw_existing]
            if isinstance(raw_existing, str)
            else (raw_existing if isinstance(raw_existing, list) else [])
        )

        for pt in pred_tags:
            if pt not in existing_tags:
                tags_to_add.setdefault(pt, []).append(pid)

        if not p.get("is_tax_curated", False):
            (
                is_tax,
                tax_conf,
                _,
                is_tax_rev,
                _,
            ) = await self.engine._classify_tax_relevance(c_vec)
            if is_tax and tax_conf >= settings.threshold_tax_confident:
                tax_confident.append(pid)
            elif is_tax_rev:
                tax_review.append(pid)

    async def _evaluate_point(
        self,
        r: Any,
        doctype_green_assign: dict[str, list[int]],
        doctype_yellow_review: list[int],
        doctype_groundtruth: list[int],
        tags_to_add: dict[str, list[int]],
        tax_confident: list[int],
        tax_review: list[int],
    ) -> bool:
        """Evaluate a single Qdrant point across doctype, tags, and tax."""
        p = getattr(r, "payload", {}) or {}
        pid_raw = p.get("paperless_id")
        if not pid_raw:
            return False
        pid = int(pid_raw)

        vec_dict = r.vector if isinstance(r.vector, dict) else {}
        raw_header = vec_dict.get("header")
        raw_content = vec_dict.get("content")

        h_vec: list[float] | None = (
            [float(x) for x in raw_header]
            if isinstance(raw_header, list) and raw_header
            else None
        )
        c_vec: list[float] | None = (
            [float(x) for x in raw_content]
            if isinstance(raw_content, list) and raw_content
            else None
        )

        await self._evaluate_doctype(
            r,
            p,
            pid,
            h_vec,
            c_vec,
            doctype_green_assign,
            doctype_yellow_review,
            doctype_groundtruth,
        )
        await self._evaluate_tags_and_tax(
            p, pid, c_vec, tags_to_add, tax_confident, tax_review
        )
        return True

    def _print_summary(
        self,
        con: Console,
        total_processed: int,
        doctype_groundtruth: list[int],
        doctype_green_assign: dict[str, list[int]],
        doctype_yellow_review: list[int],
        tags_to_add: dict[str, list[int]],
        tax_confident: list[int],
        tax_review: list[int],
    ) -> None:
        """Render Rich evaluation summary tables."""
        table = Table(
            title=f"Batch-Klassifikation Übersicht ({total_processed} Dokumente)"
        )
        table.add_column("Bereich", style="cyan")
        table.add_column("Kategorie / Status", style="bold")
        table.add_column("Anzahl Dokumente", justify="right", style="bold green")
        table.add_column("Aktion")

        table.add_row(
            "Dokumenttypen",
            "Goldstandard (Unverändert)",
            str(len(doctype_groundtruth)),
            "Bleibt fixiert",
        )
        green_total = sum(len(ids) for ids in doctype_green_assign.values())
        table.add_row(
            "Dokumenttypen",
            "Grüne Zone (Aktualisierung)",
            str(green_total),
            "Setzt neuen Dokumenttyp (Bulk-Edit)",
        )
        table.add_row(
            "Dokumenttypen",
            "Gelbe Zone (Review-Queue)",
            str(len(doctype_yellow_review)),
            "Erhält Tag 'Doctype-Review'",
        )

        total_tag_ops = sum(len(ids) for ids in tags_to_add.values())
        table.add_row(
            "Thementags",
            "Makro-Tags Zuweisungen",
            str(total_tag_ops),
            f"Über {len(tags_to_add)} Makro-Themen verteilt",
        )
        table.add_row(
            "Steuern",
            "Steuerrelevant (Hohe Konfidenz)",
            str(len(tax_confident)),
            "Erhält Tag 'steuerrelevant'",
        )
        table.add_row(
            "Steuern",
            "Steuer-Review (Prüfung)",
            str(len(tax_review)),
            "Erhält Tag 'Steuer-Review'",
        )
        con.print(table)

    async def _post_bulk_chunks(
        self,
        client: httpx.AsyncClient,
        con: Console,
        doc_ids: list[int],
        method: str,
        params: dict[str, Any],
        label: str,
    ) -> None:
        """Execute resilient bulk edit chunks of 50 documents."""
        if not doc_ids:
            return
        con.print(f"{label} ({len(doc_ids)} Dokumente)...")
        for chunk_idx in range(0, len(doc_ids), 50):
            chunk = doc_ids[chunk_idx : chunk_idx + 50]
            payload = {"documents": chunk, "method": method, "parameters": params}
            for attempt in range(3):
                try:
                    resp = await client.post(
                        f"{self.paperless_url}/api/documents/bulk_edit/",
                        json=payload,
                        timeout=httpx.Timeout(300.0, connect=60.0),
                    )
                    resp.raise_for_status()
                    await asyncio.sleep(0.3)
                    break
                except Exception as exc:
                    if attempt == 2:
                        con.print(f"[red]Failed chunk bulk_edit: {exc}[/red]")
                    else:
                        await asyncio.sleep(2.0)

    async def run(
        self,
        dry_run: bool = True,
        limit: int | None = None,
        console: Console | None = None,
    ) -> dict[str, Any]:
        """Execute full corpus batch evaluation or application."""
        con = console or Console()
        await self.engine.initialize_anchor_embeddings()

        async with httpx.AsyncClient(
            verify=settings.paperless_ssl_verify,
            headers=self._get_headers(),
            timeout=settings.paperless_timeout,
        ) as client:
            con.print("[cyan]Fetching document types and tags from Paperless...[/cyan]")
            (
                doc_type_map,
                _,
                doctype_review_tag_id,
                steuer_review_tag_id,
                steuerrelevant_tag_id,
                macro_tag_ids,
            ) = await self._load_metadata(client, dry_run)

            con.print(
                f"[cyan]Scrolling documents from Qdrant (dry_run={dry_run})...[/cyan]"
            )
            q_client = self.qdrant.client
            next_page = None
            total_processed = 0

            doctype_green_assign: dict[str, list[int]] = {}
            doctype_yellow_review: list[int] = []
            doctype_groundtruth: list[int] = []
            tags_to_add: dict[str, list[int]] = {}
            tax_confident: list[int] = []
            tax_review: list[int] = []

            while True:
                records, next_page = await q_client.scroll(
                    collection_name=self.qdrant.collection,
                    limit=200,
                    with_payload=True,
                    with_vectors=["header", "content"],
                    offset=next_page,
                )
                if not records:
                    break

                for r in records:
                    ok = await self._evaluate_point(
                        r,
                        doctype_green_assign,
                        doctype_yellow_review,
                        doctype_groundtruth,
                        tags_to_add,
                        tax_confident,
                        tax_review,
                    )
                    if ok:
                        total_processed += 1
                    if limit and total_processed >= limit:
                        break

                if (limit and total_processed >= limit) or next_page is None:
                    break

            self._print_summary(
                con,
                total_processed,
                doctype_groundtruth,
                doctype_green_assign,
                doctype_yellow_review,
                tags_to_add,
                tax_confident,
                tax_review,
            )

            if not dry_run:
                con.print(
                    "\n[bold yellow]Applying changes in Paperless via native Bulk API (Chunks of 50)...[/bold yellow]"
                )
                for dt_name, ids in doctype_green_assign.items():
                    dt_id = doc_type_map.get(dt_name)
                    if dt_id:
                        await self._post_bulk_chunks(
                            client,
                            con,
                            ids,
                            "set_document_type",
                            {"document_type": dt_id},
                            f"Set Doctype '{dt_name}'",
                        )

                await self._post_bulk_chunks(
                    client,
                    con,
                    doctype_yellow_review,
                    "modify_tags",
                    {"add_tags": [doctype_review_tag_id], "remove_tags": []},
                    "Tagging Doctype-Review",
                )

                for tname, ids in tags_to_add.items():
                    tid = macro_tag_ids.get(tname)
                    if tid:
                        await self._post_bulk_chunks(
                            client,
                            con,
                            ids,
                            "modify_tags",
                            {"add_tags": [tid], "remove_tags": []},
                            f"Tagging '{tname}'",
                        )

                await self._post_bulk_chunks(
                    client,
                    con,
                    tax_confident,
                    "modify_tags",
                    {"add_tags": [steuerrelevant_tag_id], "remove_tags": []},
                    "Tagging steuerrelevant",
                )
                await self._post_bulk_chunks(
                    client,
                    con,
                    tax_review,
                    "modify_tags",
                    {"add_tags": [steuer_review_tag_id], "remove_tags": []},
                    "Tagging Steuer-Review",
                )

                con.print(
                    "[bold green]Bulk application successfully completed in Paperless![/bold green]"
                )

            return {
                "total_processed": total_processed,
                "doctype_green": sum(len(ids) for ids in doctype_green_assign.values()),
                "doctype_yellow": len(doctype_yellow_review),
                "tags_added": sum(len(ids) for ids in tags_to_add.values()),
                "tax_confident": len(tax_confident),
                "tax_review": len(tax_review),
            }
