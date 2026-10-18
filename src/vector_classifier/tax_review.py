"""Tax review workflow: candidate tagging and curated synchronization."""

import asyncio
import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from vector_classifier.config import settings
from vector_classifier.vectors.qdrant import QdrantStorageManager

logger = logging.getLogger(__name__)

DEFAULT_PRIVAT_FILE = Path("/home/gotti/paperless-custom-classifier/steuern-privat.txt")
DEFAULT_VERMIETUNG_FILE = Path(
    "/home/gotti/paperless-custom-classifier/steuern-vermietung.txt"
)
DEFAULT_CANDIDATES_CSV = Path("tax_candidates_2026.csv")


def extract_ecodms_id(payload: dict[str, Any]) -> int | None:
    """Extract ecoDMS ID from payload filename or title."""
    fn = str(payload.get("filename", ""))
    title = str(payload.get("title", ""))
    for text in (fn, title):
        m = re.search(r"ecodms_docid_0*(\d+)", text)
        if m:
            return int(m.group(1))
        m2 = re.match(r"^(\d+)", text)
        if m2:
            return int(m2.group(1))
    return None


class TaxReviewManager:
    """Manages tagging candidates in Paperless and syncing reviewed documents into curated Qdrant ground-truth."""

    def __init__(
        self,
        paperless_url: str = settings.paperless_base_url,
        paperless_token: str = settings.paperless_api_token,
        privat_file: Path = DEFAULT_PRIVAT_FILE,
        vermietung_file: Path = DEFAULT_VERMIETUNG_FILE,
        qdrant_manager: QdrantStorageManager | None = None,
    ) -> None:
        self.paperless_url = paperless_url.rstrip("/")
        self.paperless_token = paperless_token
        self.privat_file = privat_file
        self.vermietung_file = vermietung_file
        self._qdrant_manager = qdrant_manager
        self._own_qdrant = qdrant_manager is None

    @property
    def qdrant(self) -> QdrantStorageManager:
        if self._qdrant_manager is None:
            self._qdrant_manager = QdrantStorageManager()
        return self._qdrant_manager

    async def close(self) -> None:
        if self._own_qdrant and self._qdrant_manager is not None:
            await self._qdrant_manager.close()
            self._qdrant_manager = None

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Token {self.paperless_token}",
        }

    async def get_or_create_tag(
        self,
        client: httpx.AsyncClient,
        name: str,
        color: str = "#f59e0b",
    ) -> int:
        """Find or create tag in Paperless."""
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

    async def get_tax_tag_ids(self, client: httpx.AsyncClient) -> dict[str, int | None]:
        """Get tag IDs for Steuern, Steuern-Privat, Steuern-Vermietung, Steuern-Review."""
        resp = await client.get(f"{self.paperless_url}/api/tags/?page_size=500")
        resp.raise_for_status()
        tags = resp.json().get("results", [])
        tag_map: dict[str, int | None] = {
            "Steuern": None,
            "Steuern-Privat": None,
            "Steuern-Vermietung": None,
            "Steuern-Review": None,
        }
        for t in tags:
            name = t["name"].strip()
            if name in tag_map:
                tag_map[name] = int(t["id"])
        return tag_map

    async def _load_candidate_rows(self, csv_path: Path) -> list[dict[str, Any]]:
        """Load candidate rows from CSV or fallback to Qdrant scroll."""
        if csv_path.exists():
            with csv_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return list(reader)

        candidate_rows: list[dict[str, Any]] = []
        q_client = self.qdrant.client
        next_page = None
        while True:
            records, next_page = await q_client.scroll(
                collection_name=self.qdrant.collection,
                limit=1000,
                with_payload=True,
                with_vectors=False,
                offset=next_page,
            )
            for r in records:
                p = r.payload or {}
                if p.get("tax_review_required"):
                    candidate_rows.append(
                        {
                            "paperless_id": p.get("paperless_id"),
                            "predicted_tax_category": p.get(
                                "tax_category", "steuer:werbungskosten"
                            ),
                            "title": p.get("title", ""),
                            "confidence": p.get("tax_confidence", 0.0),
                        }
                    )
            if next_page is None:
                break
        return candidate_rows

    async def tag_candidates(
        self,
        csv_path: Path = DEFAULT_CANDIDATES_CSV,
        dry_run: bool = False,
        console: Console | None = None,
    ) -> dict[str, int]:
        """Tag all candidate documents in Paperless with Steuern-Review and proposed category tag."""
        out = console or Console()
        stats = {"candidates": 0, "updated": 0, "already_tagged": 0, "errors": 0}

        candidate_rows = await self._load_candidate_rows(csv_path)
        stats["candidates"] = len(candidate_rows)
        out.print(f"Found {len(candidate_rows)} tax candidates to tag.")

        async with httpx.AsyncClient(
            headers=self._get_headers(),
            timeout=settings.paperless_timeout,
            verify=settings.paperless_ssl_verify,
        ) as client:
            review_tag_id = await self.get_or_create_tag(
                client, "Steuern-Review", color="#f59e0b"
            )
            tag_map = await self.get_tax_tag_ids(client)
            privat_tag_id = tag_map.get("Steuern-Privat")
            vermietung_tag_id = tag_map.get("Steuern-Vermietung")

            semaphore = asyncio.Semaphore(4)

            async def _process_row(row: dict[str, Any]) -> str:
                raw_pid = row.get("paperless_id")
                if not raw_pid:
                    return "skip"
                pid = int(raw_pid)
                pred_cat = str(row.get("predicted_tax_category", "")).lower()
                target_tag_id = (
                    vermietung_tag_id if "anlage-v" in pred_cat else privat_tag_id
                )

                async with semaphore:
                    try:
                        doc_resp = await client.get(
                            f"{self.paperless_url}/api/documents/{pid}/"
                        )
                        if doc_resp.status_code != 200:
                            out.print(
                                f"[red]Failed to fetch doc {pid}: HTTP {doc_resp.status_code}[/red]"
                            )
                            return "error"

                        doc_data = doc_resp.json()
                        current_tags = list(doc_data.get("tags", []))
                        new_tags = list(current_tags)

                        if review_tag_id not in new_tags:
                            new_tags.append(review_tag_id)
                        if target_tag_id and target_tag_id not in new_tags:
                            new_tags.append(target_tag_id)

                        if set(new_tags) == set(current_tags):
                            return "already_tagged"

                        if not dry_run:
                            patch_resp = await client.patch(
                                f"{self.paperless_url}/api/documents/{pid}/",
                                json={"tags": new_tags},
                            )
                            patch_resp.raise_for_status()

                        return "updated"
                    except Exception as exc:
                        out.print(f"[red]Error tagging doc {pid}: {exc}[/red]")
                        return "error"

            results = await asyncio.gather(*[_process_row(r) for r in candidate_rows])
            for res in results:
                if res == "updated":
                    stats["updated"] += 1
                elif res == "already_tagged":
                    stats["already_tagged"] += 1
                elif res == "error":
                    stats["errors"] += 1

        return stats

    async def sync_curated(
        self,
        dry_run: bool = False,
        console: Console | None = None,
    ) -> dict[str, int]:
        """Synchronize reviewed candidates from Paperless into Qdrant curated status and text files."""
        out = console or Console()
        stats = {
            "total_candidates": 0,
            "confirmed": 0,
            "rejected": 0,
            "pending": 0,
            "errors": 0,
        }

        manager = self.qdrant
        client = manager.client

        # 1. Fetch all documents in Qdrant with tax_review_required == True
        all_points: list[Any] = []
        next_page = None
        while True:
            records, next_page = await client.scroll(
                collection_name=manager.collection,
                limit=1000,
                with_payload=True,
                with_vectors=False,
                offset=next_page,
            )
            all_points.extend(records)
            if next_page is None:
                break

        candidates = [
            r for r in all_points if (r.payload or {}).get("tax_review_required")
        ]

        stats["total_candidates"] = len(candidates)
        out.print(
            f"Checking {len(candidates)} candidate documents for review completion..."
        )

        async with httpx.AsyncClient(
            headers=self._get_headers(),
            timeout=settings.paperless_timeout,
            verify=settings.paperless_ssl_verify,
        ) as http_client:
            tag_map = await self.get_tax_tag_ids(http_client)
            review_tag_id = tag_map.get("Steuern-Review")
            privat_tag_id = tag_map.get("Steuern-Privat")
            vermietung_tag_id = tag_map.get("Steuern-Vermietung")
            steuern_tag_id = tag_map.get("Steuern")

            tax_tag_ids = {
                tid
                for tid in (privat_tag_id, vermietung_tag_id, steuern_tag_id)
                if tid is not None
            }

            for r in candidates:
                p = r.payload or {}
                pid = p.get("paperless_id")
                if not pid:
                    continue

                try:
                    doc_resp = await http_client.get(
                        f"{self.paperless_url}/api/documents/{pid}/"
                    )
                    if doc_resp.status_code != 200:
                        stats["errors"] += 1
                        continue

                    doc = doc_resp.json()
                    current_tags = set(doc.get("tags", []))

                    # If review tag is still present -> still pending
                    if review_tag_id and review_tag_id in current_tags:
                        stats["pending"] += 1
                        continue

                    # If user removed review tag, check if any tax tag is present
                    has_tax = bool(current_tags & tax_tag_ids)

                    if has_tax:
                        # Confirmed!
                        is_vermietung = bool(
                            vermietung_tag_id and vermietung_tag_id in current_tags
                        )
                        category = (
                            "steuer:anlage-v"
                            if is_vermietung
                            else p.get("tax_category", "steuer:werbungskosten")
                        )
                        source_name = (
                            "steuern-vermietung" if is_vermietung else "steuern-privat"
                        )
                        patch = {
                            "is_tax_curated": True,
                            "is_tax_relevant": True,
                            "tax_review_required": False,
                            "tax_confidence": 1.0,
                            "tax_source": source_name,
                            "tax_category": category,
                        }
                        if not dry_run:
                            await client.set_payload(
                                collection_name=manager.collection,
                                payload=patch,
                                points=[r.id],
                                wait=False,
                            )
                            # Append to curated text file
                            target_file = (
                                self.vermietung_file
                                if is_vermietung
                                else self.privat_file
                            )
                            self._append_to_tax_file(target_file, doc, p)
                        stats["confirmed"] += 1
                    else:
                        # Rejected!
                        patch = {
                            "is_tax_curated": False,
                            "is_tax_relevant": False,
                            "tax_review_required": False,
                            "tax_confidence": 0.0,
                            "tax_source": None,
                            "tax_category": None,
                        }
                        if not dry_run:
                            await client.set_payload(
                                collection_name=manager.collection,
                                payload=patch,
                                points=[r.id],
                                wait=False,
                            )
                        stats["rejected"] += 1

                except Exception as exc:
                    out.print(f"[red]Error syncing doc {pid}: {exc}[/red]")
                    stats["errors"] += 1

        return stats

    def _append_to_tax_file(
        self,
        target_file: Path,
        doc: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        """Append confirmed document to curated text file if not already present."""
        if not target_file.exists():
            return

        content = target_file.read_text(encoding="utf-8")
        eid = extract_ecodms_id(payload)
        doc_id = doc.get("id")

        # Check if already present by filename ID
        existing_ids: set[int] = set()
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fname = Path(line).name
            m = re.match(r"^(\d+)", fname)
            if m:
                existing_ids.add(int(m.group(1)))

        if eid and eid in existing_ids:
            return
        if doc_id and int(doc_id) in existing_ids:
            return

        # Determine year
        created = doc.get("created", "")
        year = "2026"
        if created:
            try:
                year = str(datetime.fromisoformat(created.replace("Z", "+00:00")).year)
            except Exception:
                pass

        title_slug = re.sub(r"[^\w\-_.]", "", doc.get("title", "document"))
        id_part = f"{eid}_" if eid else f"{doc_id}_"
        line = f"./{year}/{id_part}{title_slug}.pdf\n"

        with target_file.open("a", encoding="utf-8") as f:
            f.write(line)
