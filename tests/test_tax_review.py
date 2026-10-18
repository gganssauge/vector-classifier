"""Unit tests for TaxReviewManager and tax review workflow."""

import csv
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import respx
from httpx import Response

from vector_classifier.tax_review import (
    TaxReviewManager,
    extract_ecodms_id,
)


def test_extract_ecodms_id():
    """Verify ecoDMS ID extraction patterns."""
    assert extract_ecodms_id({"filename": "ecodms_docid_0003282_rev.pdf"}) == 3282
    assert extract_ecodms_id({"title": "ecodms_docid_0000045_scan"}) == 45
    assert extract_ecodms_id({"filename": "4934_Kontoauszug.pdf"}) == 4934
    assert extract_ecodms_id({"title": "Keine ID hier"}) is None
    assert extract_ecodms_id({}) is None


@pytest.mark.asyncio
async def test_get_or_create_tag_existing():
    """Verify existing tag ID is returned without POST."""
    mgr = TaxReviewManager()
    with respx.mock(base_url="https://paperless.gorki120.de") as respx_mock:
        respx_mock.get("/api/tags/?page_size=500").mock(
            return_value=Response(
                200,
                json={"results": [{"id": 270, "name": "Steuern-Review"}]},
            )
        )
        import httpx

        async with httpx.AsyncClient() as client:
            tag_id = await mgr.get_or_create_tag(client, "Steuern-Review")
            assert tag_id == 270


@pytest.mark.asyncio
async def test_get_or_create_tag_creates_new():
    """Verify tag is created when missing."""
    mgr = TaxReviewManager()
    with respx.mock(base_url="https://paperless.gorki120.de") as respx_mock:
        respx_mock.get("/api/tags/?page_size=500").mock(
            return_value=Response(200, json={"results": []})
        )
        respx_mock.post("/api/tags/").mock(
            return_value=Response(201, json={"id": 299, "name": "Steuern-Review"})
        )
        import httpx

        async with httpx.AsyncClient() as client:
            tag_id = await mgr.get_or_create_tag(client, "Steuern-Review")
            assert tag_id == 299


@pytest.mark.asyncio
async def test_tag_candidates_dry_run(tmp_path: Path):
    """Verify candidate tagging logic in dry run."""
    csv_file = tmp_path / "candidates.csv"
    with csv_file.open("w", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["paperless_id", "predicted_tax_category", "title"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "paperless_id": "101",
                "predicted_tax_category": "steuer:anlage-v",
                "title": "Auszug",
            }
        )

    mgr = TaxReviewManager()

    with respx.mock(base_url="https://paperless.gorki120.de") as respx_mock:
        respx_mock.get("/api/tags/?page_size=500").mock(
            return_value=Response(
                200,
                json={
                    "results": [
                        {"id": 270, "name": "Steuern-Review"},
                        {"id": 248, "name": "Steuern-Privat"},
                        {"id": 250, "name": "Steuern-Vermietung"},
                    ]
                },
            )
        )
        respx_mock.get("/api/documents/101/").mock(
            return_value=Response(200, json={"id": 101, "tags": [12]})
        )

        stats = await mgr.tag_candidates(csv_path=csv_file, dry_run=True)
        assert stats["candidates"] == 1
        assert stats["updated"] == 1
        assert stats["errors"] == 0


@pytest.mark.asyncio
async def test_tag_candidates_apply(tmp_path: Path):
    """Verify candidate tagging applies PATCH to Paperless."""
    csv_file = tmp_path / "candidates.csv"
    with csv_file.open("w", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["paperless_id", "predicted_tax_category", "title"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "paperless_id": "102",
                "predicted_tax_category": "steuer:werbungskosten",
                "title": "Rechnung",
            }
        )

    mgr = TaxReviewManager()

    with respx.mock(base_url="https://paperless.gorki120.de") as respx_mock:
        respx_mock.get("/api/tags/?page_size=500").mock(
            return_value=Response(
                200,
                json={
                    "results": [
                        {"id": 270, "name": "Steuern-Review"},
                        {"id": 248, "name": "Steuern-Privat"},
                    ]
                },
            )
        )
        respx_mock.get("/api/documents/102/").mock(
            return_value=Response(200, json={"id": 102, "tags": []})
        )
        patch_route = respx_mock.patch("/api/documents/102/").mock(
            return_value=Response(200, json={"id": 102, "tags": [270, 248]})
        )

        stats = await mgr.tag_candidates(csv_path=csv_file, dry_run=False)
        assert stats["candidates"] == 1
        assert stats["updated"] == 1
        assert patch_route.called


@pytest.mark.asyncio
async def test_sync_curated_workflow(tmp_path: Path):
    """Verify sync_curated handles pending, confirmed, and rejected documents."""
    privat_txt = tmp_path / "privat.txt"
    vermietung_txt = tmp_path / "vermietung.txt"
    privat_txt.write_text("./2021/3000_alt.pdf\n", encoding="utf-8")
    vermietung_txt.write_text("./2021/4000_alt.pdf\n", encoding="utf-8")

    mock_qdrant = MagicMock()
    point_pending = MagicMock(
        id="pt-1",
        payload={"paperless_id": 201, "tax_review_required": True, "title": "Doc1"},
    )
    point_confirmed = MagicMock(
        id="pt-2",
        payload={
            "paperless_id": 202,
            "tax_review_required": True,
            "title": "ecodms_docid_0005555_Doc2",
        },
    )
    point_rejected = MagicMock(
        id="pt-3",
        payload={"paperless_id": 203, "tax_review_required": True, "title": "Doc3"},
    )

    mock_qdrant.client.scroll = AsyncMock(
        return_value=([point_pending, point_confirmed, point_rejected], None)
    )
    mock_qdrant.client.set_payload = AsyncMock()
    mock_qdrant.collection = "paperless_documents"

    mgr = TaxReviewManager(
        privat_file=privat_txt,
        vermietung_file=vermietung_txt,
        qdrant_manager=mock_qdrant,
    )

    with respx.mock(base_url="https://paperless.gorki120.de") as respx_mock:
        respx_mock.get("/api/tags/?page_size=500").mock(
            return_value=Response(
                200,
                json={
                    "results": [
                        {"id": 270, "name": "Steuern-Review"},
                        {"id": 248, "name": "Steuern-Privat"},
                        {"id": 250, "name": "Steuern-Vermietung"},
                    ]
                },
            )
        )
        # Doc 201: Still has Steuern-Review (270) -> PENDING
        respx_mock.get("/api/documents/201/").mock(
            return_value=Response(200, json={"id": 201, "tags": [270, 248]})
        )
        # Doc 202: User removed 270, kept 250 (Vermietung) -> CONFIRMED VERMIETUNG
        respx_mock.get("/api/documents/202/").mock(
            return_value=Response(
                200,
                json={
                    "id": 202,
                    "title": "Reparatur",
                    "tags": [250],
                    "created": "2026-03-01T12:00:00Z",
                },
            )
        )
        # Doc 203: User removed 270 and has NO tax tags -> REJECTED
        respx_mock.get("/api/documents/203/").mock(
            return_value=Response(200, json={"id": 203, "tags": [12]})
        )

        stats = await mgr.sync_curated(dry_run=False)
        assert stats["total_candidates"] == 3
        assert stats["pending"] == 1
        assert stats["confirmed"] == 1
        assert stats["rejected"] == 1

        # Verify Vermietung text file was appended
        vermietung_content = vermietung_txt.read_text(encoding="utf-8")
        assert "5555" in vermietung_content

        # Verify Qdrant payload was updated for confirmed & rejected
        assert mock_qdrant.client.set_payload.call_count == 2
        rejected_patch = mock_qdrant.client.set_payload.call_args_list[1].kwargs[
            "payload"
        ]
        assert rejected_patch["is_tax_curated"] is True
        assert rejected_patch["is_tax_relevant"] is False
        assert rejected_patch["tax_source"] == "confirmed_nontax"
        rejected_patch = mock_qdrant.client.set_payload.call_args_list[1].kwargs[
            "payload"
        ]
        assert rejected_patch["is_tax_curated"] is True
        assert rejected_patch["is_tax_relevant"] is False
        assert rejected_patch["tax_source"] == "confirmed_nontax"
