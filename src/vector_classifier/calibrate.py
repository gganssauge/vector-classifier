"""Three-zone tax relevance calibration and candidate extraction."""

import csv
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from vector_classifier.config import settings
from vector_classifier.tax_review import (
    DEFAULT_CANDIDATES_CSV,
    DEFAULT_PRIVAT_FILE,
    DEFAULT_VERMIETUNG_FILE,
    extract_ecodms_id,
)
from vector_classifier.vectors.qdrant import QdrantStorageManager

logger = logging.getLogger(__name__)


def parse_tax_list(path: Path) -> dict[int, str]:
    """Parse list of ecoDMS doc IDs from curated file."""
    result: dict[int, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        filename = Path(line).name
        m = re.match(r"^(\d+)", filename)
        if m:
            result[int(m.group(1))] = filename
    return result


class TaxCalibrationRunner:
    """Executes 3-zone tax relevance calibration across Qdrant."""

    def __init__(
        self,
        privat_file: Path = DEFAULT_PRIVAT_FILE,
        vermietung_file: Path = DEFAULT_VERMIETUNG_FILE,
        candidates_csv: Path = DEFAULT_CANDIDATES_CSV,
        threshold_knn: float = 0.82,
        qdrant_manager: QdrantStorageManager | None = None,
    ) -> None:
        self.privat_file = privat_file
        self.vermietung_file = vermietung_file
        self.candidates_csv = candidates_csv
        self.threshold_knn = threshold_knn
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

    async def fetch_paperless_metadata(self) -> dict[int, dict[str, Any]]:
        """Fetch doc_id -> metadata from Paperless API."""
        metadata: dict[int, dict[str, Any]] = {}
        headers = {"Authorization": f"Token {settings.paperless_api_token}"}
        async with httpx.AsyncClient(
            headers=headers,
            timeout=settings.paperless_timeout,
            verify=settings.paperless_ssl_verify,
        ) as client:
            page = 1
            while True:
                resp = await client.get(
                    f"{settings.paperless_base_url}/api/documents/?page={page}&page_size=250&fields=id,created,title,custom_fields"
                )
                if resp.status_code != 200:
                    break
                data = resp.json()
                for doc in data.get("results", []):
                    doc_id = doc.get("id")
                    created_str = doc.get("created", "")
                    year = None
                    if created_str:
                        try:
                            year = datetime.fromisoformat(
                                created_str.replace("Z", "+00:00")
                            ).year
                        except Exception:
                            pass

                    ecodms_id = None
                    for cf in doc.get("custom_fields", []):
                        if cf.get("field") == 1 and cf.get("value") is not None:
                            try:
                                ecodms_id = int(cf["value"])
                            except Exception:
                                pass
                        if cf.get("field") == 9 and cf.get("value") is not None:
                            try:
                                year = int(cf["value"])
                            except Exception:
                                pass

                    metadata[doc_id] = {
                        "year": year,
                        "ecodms_id": ecodms_id,
                        "created": created_str,
                        "title": doc.get("title", ""),
                    }
                if not data.get("next"):
                    break
                page += 1
        return metadata

    async def run(
        self,
        console: Console | None = None,
    ) -> dict[str, Any]:
        """Run 3-zone calibration and produce candidates CSV."""
        out = console or Console()
        out.print("[bold cyan]Starting 3-Zone Tax Relevance Calibration...[/bold cyan]")

        privat_ids = parse_tax_list(self.privat_file)
        vermietung_ids = parse_tax_list(self.vermietung_file)

        paperless_meta = await self.fetch_paperless_metadata()
        out.print(f"Loaded metadata for {len(paperless_meta)} Paperless docs.")

        manager = self.qdrant
        client = manager.client

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

        stats = {
            "curated_2020_2025": 0,
            "nontax_2020_2025": 0,
            "historical_pre_2020_tax": 0,
            "historical_pre_2020_nontax": 0,
            "post_2025_total": 0,
            "post_2025_tax_candidates": 0,
        }

        updates: list[tuple[Any, dict[str, Any]]] = []
        post_2025_points: list[tuple[Any, Any, str, int | None, int | None]] = []

        for r in all_points:
            p = r.payload or {}
            pid = p.get("paperless_id")
            meta = paperless_meta.get(pid, {})
            year = meta.get("year")
            eid = meta.get("ecodms_id") or extract_ecodms_id(p)
            tags_orig = [str(t).lower() for t in p.get("tags_original", [])]
            has_paperless_tax_tag = any("steuer" in t for t in tags_orig)

            # Zone 1: 2020 .. 2025
            if year is not None and 2020 <= year <= 2025:
                if eid in vermietung_ids:
                    stats["curated_2020_2025"] += 1
                    patch = {
                        "doc_year": year,
                        "ecodms_id": eid,
                        "is_tax_curated": True,
                        "is_tax_relevant": True,
                        "tax_review_required": False,
                        "tax_source": "steuern-vermietung",
                        "tax_category": "steuer:anlage-v",
                        "tax_confidence": 1.0,
                    }
                elif eid in privat_ids:
                    stats["curated_2020_2025"] += 1
                    orig_cat = p.get("tax_category")
                    cat = (
                        orig_cat
                        if orig_cat
                        and "krankheit" not in orig_cat
                        and "anlage-v" not in orig_cat
                        else "steuer:werbungskosten"
                    )
                    patch = {
                        "doc_year": year,
                        "ecodms_id": eid,
                        "is_tax_curated": True,
                        "is_tax_relevant": True,
                        "tax_review_required": False,
                        "tax_source": "steuern-privat",
                        "tax_category": cat,
                        "tax_confidence": 1.0,
                    }
                else:
                    stats["nontax_2020_2025"] += 1
                    patch = {
                        "doc_year": year,
                        "ecodms_id": eid,
                        "is_tax_curated": False,
                        "is_tax_relevant": False,
                        "tax_review_required": False,
                        "tax_source": None,
                        "tax_category": None,
                        "tax_confidence": 0.0,
                    }
                updates.append((r.id, patch))

            # Zone 2: Pre-2020 (< 2020)
            elif year is not None and year < 2020:
                if has_paperless_tax_tag:
                    stats["historical_pre_2020_tax"] += 1
                    cat = (
                        "steuer:anlage-v"
                        if any("vermietung" in t for t in tags_orig)
                        else "steuer:werbungskosten"
                    )
                    patch = {
                        "doc_year": year,
                        "ecodms_id": eid,
                        "is_tax_curated": False,
                        "is_tax_relevant": True,
                        "tax_review_required": False,
                        "tax_source": "paperless_tag",
                        "tax_category": cat,
                        "tax_confidence": 1.0,
                    }
                else:
                    stats["historical_pre_2020_nontax"] += 1
                    patch = {
                        "doc_year": year,
                        "ecodms_id": eid,
                        "is_tax_curated": False,
                        "is_tax_relevant": False,
                        "tax_review_required": False,
                        "tax_source": None,
                        "tax_category": None,
                        "tax_confidence": 0.0,
                    }
                updates.append((r.id, patch))

            # Zone 3: Post-2025 (2026+) or unknown
            else:
                stats["post_2025_total"] += 1
                post_2025_points.append(
                    (r.id, pid, meta.get("title", p.get("title", "")), year, eid)
                )

        # Apply Zone 1 & 2 updates to Qdrant
        for i in range(0, len(updates), 100):
            for pt_id, patch in updates[i : i + 100]:
                await client.set_payload(
                    collection_name=manager.collection,
                    payload=patch,
                    points=[pt_id],
                    wait=False,
                )

        # Zone 3: Evaluate 2026+ via k-NN against curated anchors
        candidates_for_review: list[dict[str, Any]] = []
        for pt_id, pid, title, yr, eid in post_2025_points:
            recs = await client.retrieve(
                collection_name=manager.collection,
                ids=[pt_id],
                with_vectors=["content"],
                with_payload=True,
            )
            pt_vector = recs[0].vector
            if not isinstance(pt_vector, dict) or "content" not in pt_vector:
                continue
            raw_vec = pt_vector["content"]
            if (
                not isinstance(raw_vec, list)
                or not raw_vec
                or isinstance(raw_vec[0], list)
            ):
                continue
            vec: list[float] = [
                float(x) for x in raw_vec if isinstance(x, (int, float))
            ]

            payload = recs[0].payload or {}
            orig_tags = payload.get("tags_original", [])

            tax_matches = await manager.search_by_vector(
                vector_name="content",
                vector=vec,
                limit=1,
                score_threshold=self.threshold_knn,
                filter_criteria={"is_tax_curated": True},
            )

            if tax_matches:
                best = tax_matches[0]
                score = round(float(best.score), 4)
                cat = best.payload.get("tax_category", "steuer:werbungskosten")
                matched_anchor_title = best.payload.get("title", "")
                stats["post_2025_tax_candidates"] += 1

                patch = {
                    "doc_year": yr,
                    "ecodms_id": eid,
                    "is_tax_curated": False,
                    "is_tax_relevant": True,
                    "tax_review_required": True,
                    "tax_source": "knn_candidate",
                    "tax_category": cat,
                    "tax_confidence": score,
                    "matched_curated_doc": matched_anchor_title,
                }
                candidates_for_review.append(
                    {
                        "paperless_id": pid,
                        "year": yr,
                        "title": title,
                        "predicted_tax_category": cat,
                        "confidence": score,
                        "matched_anchor_document": matched_anchor_title,
                        "current_paperless_tags": ", ".join(orig_tags),
                    }
                )
            else:
                patch = {
                    "doc_year": yr,
                    "ecodms_id": eid,
                    "is_tax_curated": False,
                    "is_tax_relevant": False,
                    "tax_review_required": False,
                    "tax_source": None,
                    "tax_category": None,
                    "tax_confidence": 0.0,
                }
            await client.set_payload(
                collection_name=manager.collection,
                payload=patch,
                points=[pt_id],
                wait=False,
            )

        with self.candidates_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "paperless_id",
                    "year",
                    "title",
                    "predicted_tax_category",
                    "confidence",
                    "matched_anchor_document",
                    "current_paperless_tags",
                ],
            )
            writer.writeheader()
            writer.writerows(candidates_for_review)

        return stats
