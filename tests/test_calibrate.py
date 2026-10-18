"""Unit tests for TaxCalibrationRunner and 3-zone tax calibration logic."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import respx
from httpx import Response

from vector_classifier.calibrate import (
    DEFAULT_TAX_CATEGORY,
    TAX_CATEGORY_ANLAGE_V,
    TaxCalibrationRunner,
    _build_zone_1_patch,
    _build_zone_2_patch,
    _build_zone_3_match,
    _build_zone_3_nontax_patch,
    _extract_point_vector,
    _parse_doc_metadata,
    _partition_points,
    _write_candidates_csv,
    parse_tax_list,
)
from vector_classifier.vectors.qdrant import QdrantStorageManager


def test_parse_tax_list_nonexistent(tmp_path: Path):
    """Missing file should return an empty dict."""
    res = parse_tax_list(tmp_path / "does_not_exist.txt")
    assert res == {}


def test_parse_tax_list_valid(tmp_path: Path):
    """Verify parsing valid lines, comments, and empty lines."""
    f = tmp_path / "test_tax.txt"
    f.write_text(
        "# Comment header\n\n./2022/4559_Rechnung_Gas.pdf\n./2023/1234_Wartung.pdf\ninvalid_line_no_id.pdf\n",
        encoding="utf-8",
    )
    res = parse_tax_list(f)
    assert res == {4559: "4559_Rechnung_Gas.pdf", 1234: "1234_Wartung.pdf"}


def test_parse_doc_metadata():
    """Verify metadata extraction from Paperless document dict."""
    doc = {
        "id": 999,
        "created": "2024-05-10T14:30:00Z",
        "title": "BSR Gebühren",
        "custom_fields": [
            {"field": 1, "value": "4020"},
            {"field": 9, "value": "2024"},
        ],
    }
    doc_id, meta = _parse_doc_metadata(doc)
    assert doc_id == 999
    assert meta["year"] == 2024
    assert meta["ecodms_id"] == 4020
    assert meta["title"] == "BSR Gebühren"


def test_build_zone_1_patch():
    """Verify Zone 1 (2020-2025) patch logic."""
    stats = {"curated_2020_2025": 0, "nontax_2020_2025": 0}
    vermietung = {100: "100.pdf"}
    privat = {200: "200.pdf"}

    # Vermietung
    p1 = _build_zone_1_patch(100, 2022, privat, vermietung, None, stats)
    assert p1["is_tax_relevant"] is True
    assert p1["tax_category"] == TAX_CATEGORY_ANLAGE_V
    assert stats["curated_2020_2025"] == 1

    # Privat
    p2 = _build_zone_1_patch(
        200, 2023, privat, vermietung, "steuer:werbungskosten", stats
    )
    assert p2["is_tax_relevant"] is True
    assert p2["tax_category"] == "steuer:werbungskosten"
    assert stats["curated_2020_2025"] == 2

    # Non-tax
    p3 = _build_zone_1_patch(300, 2021, privat, vermietung, None, stats)
    assert p3["is_tax_relevant"] is False
    assert stats["nontax_2020_2025"] == 1


def test_build_zone_2_patch():
    """Verify Zone 2 (< 2020) historical patch logic."""
    stats = {"historical_pre_2020_tax": 0, "historical_pre_2020_nontax": 0}

    # Historical tax with vermietung tag
    p1 = _build_zone_2_patch(50, 2018, True, ["steuern", "steuern-vermietung"], stats)
    assert p1["is_tax_relevant"] is True
    assert p1["tax_category"] == TAX_CATEGORY_ANLAGE_V
    assert stats["historical_pre_2020_tax"] == 1

    # Historical tax privat
    p2 = _build_zone_2_patch(51, 2019, True, ["steuern-privat"], stats)
    assert p2["is_tax_relevant"] is True
    assert p2["tax_category"] == DEFAULT_TAX_CATEGORY
    assert stats["historical_pre_2020_tax"] == 2

    # Historical non-tax
    p3 = _build_zone_2_patch(52, 2017, False, ["kontoauszug"], stats)
    assert p3["is_tax_relevant"] is False
    assert stats["historical_pre_2020_nontax"] == 1


def test_extract_point_vector():
    """Verify vector extraction from Qdrant retrieve records."""
    assert _extract_point_vector([]) is None

    mock_rec = MagicMock()
    mock_rec.vector = {"content": [0.1, 0.2, 0.3]}
    assert _extract_point_vector([mock_rec]) == [0.1, 0.2, 0.3]

    mock_invalid = MagicMock()
    mock_invalid.vector = "invalid"
    assert _extract_point_vector([mock_invalid]) is None


def test_build_zone_3_match_and_write_csv(tmp_path: Path):
    """Verify Zone 3 match builder and candidate CSV writer."""
    stats = {"post_2025_tax_candidates": 0}
    mock_best = MagicMock()
    mock_best.score = 0.885
    mock_best.payload = {
        "tax_category": "steuer:werbungskosten",
        "title": "Vorjahresbeleg",
    }

    patch, row = _build_zone_3_match(
        best=mock_best,
        yr=2026,
        eid=4900,
        pid=1001,
        title="Internet 2026",
        orig_tags=["Telekom"],
        stats=stats,
        review_reason="Test Match",
        nontax_score=0.72,
    )
    assert patch["is_tax_relevant"] is True
    assert patch["tax_confidence"] == 0.885
    assert row["paperless_id"] == 1001
    assert stats["post_2025_tax_candidates"] == 1

    nontax_patch = _build_zone_3_nontax_patch(yr=2026, eid=4900)
    assert nontax_patch["is_tax_relevant"] is False

    csv_file = tmp_path / "candidates.csv"
    _write_candidates_csv(csv_file, [row])
    assert csv_file.exists()
    assert "Internet 2026" in csv_file.read_text(encoding="utf-8")


def test_partition_points():
    """Verify partitioning across the 3 temporal zones."""
    stats = {
        "curated_2020_2025": 0,
        "nontax_2020_2025": 0,
        "historical_pre_2020_tax": 0,
        "historical_pre_2020_nontax": 0,
        "post_2025_total": 0,
        "post_2025_tax_candidates": 0,
    }

    r1 = MagicMock(id="p1", payload={"paperless_id": 1, "tags_original": ["steuern"]})
    r2 = MagicMock(id="p2", payload={"paperless_id": 2, "tags_original": ["privat"]})
    r3 = MagicMock(id="p3", payload={"paperless_id": 3, "title": "2026 Doc"})

    paperless_meta = {
        1: {"year": 2022, "ecodms_id": 10},
        2: {"year": 2018, "ecodms_id": 20},
        3: {"year": 2026, "ecodms_id": 30, "title": "2026 Doc"},
    }

    updates, post_2025 = _partition_points(
        all_points=[r1, r2, r3],
        paperless_meta=paperless_meta,
        privat_ids={10: "10.pdf"},
        vermietung_ids={},
        stats=stats,
    )

    assert len(updates) == 2
    assert len(post_2025) == 1
    assert stats["post_2025_total"] == 1


@pytest.mark.asyncio
async def test_calibration_runner_end_to_end(tmp_path: Path):
    """Verify TaxCalibrationRunner run execution with mocks."""
    privat_f = tmp_path / "privat.txt"
    vermietung_f = tmp_path / "vermietung.txt"
    candidates_csv = tmp_path / "candidates.csv"
    privat_f.write_text("./2022/100_doc.pdf\n", encoding="utf-8")
    vermietung_f.write_text("./2022/200_doc.pdf\n", encoding="utf-8")

    mock_qdrant = MagicMock(spec=QdrantStorageManager)
    mock_client = AsyncMock()
    mock_qdrant.client = mock_client
    mock_qdrant.collection = "test_collection"

    # Mock scroll returning 1 point from 2026
    point_2026 = MagicMock()
    point_2026.id = "uuid-2026"
    point_2026.payload = {
        "paperless_id": 500,
        "is_tax_curated": False,
        "title": "Wartung",
    }
    point_2026.vector = {"content": [0.1] * 1024}

    mock_client.scroll.return_value = ([point_2026], None)
    mock_client.retrieve.return_value = [point_2026]

    # Mock search_by_vector for tax and nontax
    mock_tax_match = MagicMock(score=0.88)
    mock_tax_match.id = "other-uuid"
    mock_tax_match.payload = {
        "tax_category": "steuer:anlage-v",
        "title": "Alte Wartung",
    }
    mock_qdrant.search_by_vector = AsyncMock(side_effect=[[mock_tax_match], []])

    runner = TaxCalibrationRunner(
        privat_file=privat_f,
        vermietung_file=vermietung_f,
        candidates_csv=candidates_csv,
        qdrant_manager=mock_qdrant,
    )

    with respx.mock(base_url="https://paperless.gorki120.de") as respx_mock:
        respx_mock.get(
            url__startswith="https://paperless.gorki120.de/api/documents/"
        ).mock(
            return_value=Response(
                200,
                json={
                    "results": [
                        {
                            "id": 500,
                            "created": "2026-02-01T10:00:00Z",
                            "title": "Wartung",
                            "custom_fields": [],
                        }
                    ],
                    "next": None,
                },
            )
        )

        stats = await runner.run()
        assert stats["post_2025_total"] == 1
        assert stats["post_2025_tax_candidates"] == 1
        assert candidates_csv.exists()
