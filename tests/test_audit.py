"""Unit tests for Batch-Indexer & Offline-Audit-Runner."""

import csv
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
from typer.testing import CliRunner

from vector_classifier.audit import (
    AUDIT_CSV_COLUMNS,
    AuditRunner,
    determine_tax_action,
    extract_current_tags,
    is_tax_tagged,
    load_documents,
    load_documents_from_file,
    run_audit,
)
from vector_classifier.cli import app
from vector_classifier.models.schemas import AuditDocument, VectorClassificationResult
from vector_classifier.vectors.classifier import VectorClassifierEngine
from vector_classifier.vectors.qdrant import QdrantStorageManager

cli_runner = CliRunner()


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------
def test_determine_tax_action():
    """Verify three-zone threshold action mapping."""
    assert determine_tax_action(0.95) == "auto-tag"
    assert determine_tax_action(0.60) == "auto-tag"
    assert determine_tax_action(0.599) == "review"
    assert determine_tax_action(0.52) == "review"
    assert determine_tax_action(0.519) == "none"
    assert determine_tax_action(0.10) == "none"


def test_extract_current_tags():
    """Verify robust extraction of tags from varied structures."""
    # List of strings
    assert extract_current_tags({"tags": ["Rechnung", "Telekom"]}) == [
        "Rechnung",
        "Telekom",
    ]
    # List of dicts
    assert extract_current_tags({"tags_original": [{"name": "Versicherung"}]}) == [
        "Versicherung"
    ]
    # Comma-separated string
    assert extract_current_tags({"current_tags": "Fahrzeug, Auto"}) == [
        "Fahrzeug",
        "Auto",
    ]
    # None or empty
    assert extract_current_tags({}) == []


def test_is_tax_tagged():
    """Verify detection of existing tax tags or flags."""
    # Explicit boolean flag
    assert is_tax_tagged({"is_currently_tax_tagged": True}, []) is True
    assert is_tax_tagged({"is_tax_relevant": True}, []) is True
    assert is_tax_tagged({"is_currently_tax_tagged": False}, ["Normal"]) is False

    # Detection from tag strings
    assert is_tax_tagged({}, ["Steuerrechnung"]) is True
    assert is_tax_tagged({}, ["TAX-2025"]) is True
    assert is_tax_tagged({}, ["Handwerkerleistungen"]) is True
    assert is_tax_tagged({}, ["Werbungskosten"]) is True
    assert is_tax_tagged({}, ["Telekommunikation", "Privat"]) is False


def test_load_documents_from_file(tmp_path: Path):
    """Verify loading documents from JSON, JSONL, and API response formats."""
    # JSONL file
    jsonl_path = tmp_path / "docs.jsonl"
    jsonl_path.write_text(
        '{"id": 1, "content": "Doc 1"}\n{"id": 2, "content": "Doc 2"}\n',
        encoding="utf-8",
    )
    docs_jsonl = load_documents_from_file(jsonl_path)
    assert len(docs_jsonl) == 2
    assert docs_jsonl[0]["id"] == 1

    # JSON Array file
    json_path = tmp_path / "docs.json"
    json_path.write_text(
        json.dumps([{"id": 10}, {"id": 20}, {"id": 30}]),
        encoding="utf-8",
    )
    docs_json = load_documents_from_file(json_path, limit=2)
    assert len(docs_json) == 2
    assert docs_json[1]["id"] == 20

    # Paperless API wrapper {"results": [...]}
    api_path = tmp_path / "paperless_api.json"
    api_path.write_text(
        json.dumps({"count": 1, "results": [{"id": 100}]}),
        encoding="utf-8",
    )
    docs_api = load_documents_from_file(api_path)
    assert len(docs_api) == 1
    assert docs_api[0]["id"] == 100

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        load_documents_from_file(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# AuditRunner Tests with Mocked Classifier Engine
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_documents_jsonl(tmp_path: Path) -> Path:
    """Create a sample JSONL test file with diverse documents."""
    data = [
        {
            "id": 101,
            "original_file_name": "telekom_rechnung.pdf",
            "content": "Deutsche Telekom Rechnung Festnetz Internet Betrag: 49.99 EUR",
            "document_type": "Rechnung",
            "tags": ["Telekommunikation", "Steuer"],
        },
        {
            "id": 102,
            "filename": "mietvertrag_2024.pdf",
            "content": "Mietvertrag über Wohnräume in Berlin. Kaltmiete 800 EUR",
            "document_type": "Vertrag",
            "tags": ["Wohnung"],
        },
        {
            "id": 103,
            "original_file_name": "it_fachliteratur.pdf",
            "content": "Fachbuch Programmierung Softwarearchitektur O'Reilly Rechnung",
            "document_type": "Rechnung",
            "tags": ["IT"],
            # Untagged for tax in legacy data, but predicted tax relevant (new discovery)
        },
    ]
    file_path = tmp_path / "sample_docs.jsonl"
    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return file_path


@pytest.mark.asyncio
async def test_audit_runner_execution_and_csv(
    sample_documents_jsonl: Path, tmp_path: Path
):
    """Test full AuditRunner flow, CSV column validity, and summary statistics."""
    mock_engine = MagicMock()
    mock_engine.initialize_anchor_embeddings = AsyncMock()
    mock_engine.close = AsyncMock()

    # Define classification return values for each of the 3 documents
    results = [
        VectorClassificationResult(
            predicted_doc_type="Rechnung",
            doc_type_confidence=0.92,
            predicted_tags=["Telekommunikation"],
            tag_scores={"Telekommunikation": 0.88},
            is_tax_relevant=True,
            tax_confidence=0.85,
            tax_category="werbungskosten",
            tax_review_required=False,
            header_vector=[0.1] * 1024,
            content_vector=[0.2] * 1024,
        ),
        VectorClassificationResult(
            predicted_doc_type="Vertrag",
            doc_type_confidence=0.89,
            predicted_tags=["Wohnung"],
            tag_scores={"Wohnung": 0.85},
            is_tax_relevant=False,
            tax_confidence=0.45,
            tax_category=None,
            tax_review_required=False,
            header_vector=[0.1] * 1024,
            content_vector=[0.2] * 1024,
        ),
        VectorClassificationResult(
            predicted_doc_type="Rechnung",
            doc_type_confidence=0.94,
            predicted_tags=["IT", "Fachliteratur"],
            tag_scores={"IT": 0.90, "Fachliteratur": 0.84},
            is_tax_relevant=True,
            tax_confidence=0.55,
            tax_category="werbungskosten",
            tax_review_required=True,
            header_vector=[0.1] * 1024,
            content_vector=[0.2] * 1024,
        ),
    ]

    mock_engine.classify = AsyncMock(side_effect=results)

    output_csv = tmp_path / "reports" / "audit_output.csv"
    runner = AuditRunner(engine=mock_engine, concurrency=2)

    records, summary = await runner.run(
        source=sample_documents_jsonl,
        output_csv=output_csv,
    )

    # Verify Summary Metrics
    assert summary.total_documents == 3
    assert summary.doc_type_evaluated == 3
    assert summary.doc_type_matches == 3
    assert summary.doc_type_accuracy == 1.0
    assert summary.total_assigned_tags == 4  # 1 + 1 + 2
    assert (
        summary.newly_discovered_tax_docs == 1
    )  # doc 103 was not currently tax tagged
    assert summary.confident_count == 1  # doc 101 has 0.85 >= 0.60 -> auto-tag
    assert summary.review_count == 1  # doc 103 has 0.55 (0.52..0.60) -> review
    assert summary.none_count == 1  # doc 102 has 0.45 < 0.52 -> none

    # Verify CSV file output
    assert output_csv.exists()
    with output_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    # Check all required columns exist
    assert reader.fieldnames is not None
    for col in AUDIT_CSV_COLUMNS:
        assert col in reader.fieldnames

    # Verify Row 1 details
    row1 = rows[0]
    assert row1["paperless_id"] == "101"
    assert row1["filename"] == "telekom_rechnung.pdf"
    assert row1["current_doc_type"] == "Rechnung"
    assert row1["predicted_doc_type"] == "Rechnung"
    assert float(row1["doc_type_confidence"]) == 0.92
    assert row1["doc_type_match"] == "True"
    assert "Telekommunikation" in row1["current_tags"]
    assert "Steuer" in row1["current_tags"]
    assert row1["predicted_tags"] == "Telekommunikation"
    assert row1["is_currently_tax_tagged"] == "True"
    assert row1["predicted_tax_relevant"] == "True"
    assert float(row1["tax_confidence"]) == 0.85
    assert row1["tax_category"] == "werbungskosten"
    assert row1["tax_action"] == "auto-tag"

    # Verify Row 3 details (newly discovered tax doc)
    row3 = rows[2]
    assert row3["paperless_id"] == "103"
    assert row3["is_currently_tax_tagged"] == "False"
    assert row3["predicted_tax_relevant"] == "True"
    assert row3["tax_action"] == "review"


@pytest.mark.asyncio
async def test_audit_runner_limit_parameter(
    sample_documents_jsonl: Path, tmp_path: Path
):
    """Verify limit parameter properly restricts document count."""
    mock_engine = MagicMock()
    mock_engine.initialize_anchor_embeddings = AsyncMock()
    mock_engine.close = AsyncMock()
    mock_engine.classify = AsyncMock(
        return_value=VectorClassificationResult(
            predicted_doc_type="Rechnung",
            doc_type_confidence=0.88,
        )
    )

    runner = AuditRunner(engine=mock_engine, concurrency=2)
    output_csv = tmp_path / "limit_output.csv"

    records, summary = await runner.run(
        source=sample_documents_jsonl,
        output_csv=output_csv,
        limit=2,
    )

    assert len(records) == 2
    assert summary.total_documents == 2
    assert mock_engine.classify.call_count == 2


@pytest.mark.asyncio
async def test_audit_runner_with_qdrant_indexing(
    sample_documents_jsonl: Path, tmp_path: Path
):
    """Verify optional Qdrant indexing when index=True."""
    mock_engine = MagicMock()
    mock_engine.initialize_anchor_embeddings = AsyncMock()
    mock_engine.close = AsyncMock()
    mock_engine.classify = AsyncMock(
        return_value=VectorClassificationResult(
            predicted_doc_type="Rechnung",
            doc_type_confidence=0.90,
            predicted_tags=["IT"],
            header_vector=[0.1] * 1024,
            content_vector=[0.2] * 1024,
        )
    )

    mock_qdrant = MagicMock()
    mock_qdrant.ensure_collection = AsyncMock()
    mock_qdrant.upsert_document = AsyncMock()
    mock_qdrant.close = AsyncMock()

    runner = AuditRunner(
        engine=mock_engine,
        qdrant_manager=mock_qdrant,
        concurrency=1,
        index=True,
        dry_run=False,
    )

    records, summary = await runner.run(
        source=sample_documents_jsonl,
        limit=2,
    )

    assert mock_qdrant.ensure_collection.call_count == 1
    assert mock_qdrant.upsert_document.call_count == 2
    # Verify call parameters
    call_args = mock_qdrant.upsert_document.call_args[1]
    assert "point_id" in call_args
    assert "header_vector" in call_args
    assert "content_vector" in call_args
    assert "payload" in call_args
    assert call_args["payload"]["source_system"] == "paperless"


@pytest.mark.asyncio
async def test_run_audit_helper(sample_documents_jsonl: Path, tmp_path: Path):
    """Verify run_audit helper executes and cleans up."""
    mock_engine = MagicMock()
    mock_engine.initialize_anchor_embeddings = AsyncMock()
    mock_engine.close = AsyncMock()
    mock_engine.classify = AsyncMock(
        return_value=VectorClassificationResult(
            predicted_doc_type="Rechnung",
            doc_type_confidence=0.85,
        )
    )

    records, summary = await run_audit(
        source=sample_documents_jsonl,
        output_csv=tmp_path / "helper_audit.csv",
        limit=1,
        engine=mock_engine,
    )

    assert len(records) == 1
    assert summary.total_documents == 1


# ---------------------------------------------------------------------------
# CLI Command Tests
# ---------------------------------------------------------------------------
def test_cli_audit_command(sample_documents_jsonl: Path, tmp_path: Path):
    """Test CLI audit command invocation via Typer CliRunner."""
    output_csv = tmp_path / "cli_audit.csv"

    mock_res = VectorClassificationResult(
        predicted_doc_type="Rechnung",
        doc_type_confidence=0.91,
        predicted_tags=["Telekommunikation"],
        tag_scores={"Telekommunikation": 0.85},
        is_tax_relevant=True,
        tax_confidence=0.86,
        tax_category="werbungskosten",
    )

    with (
        patch(
            "vector_classifier.audit.VectorClassifierEngine.initialize_anchor_embeddings",
            new_callable=AsyncMock,
        ),
        patch(
            "vector_classifier.audit.VectorClassifierEngine.classify",
            new_callable=AsyncMock,
            return_value=mock_res,
        ),
        patch(
            "vector_classifier.audit.VectorClassifierEngine.close",
            new_callable=AsyncMock,
        ),
    ):
        result = cli_runner.invoke(
            app,
            [
                "audit",
                "--source",
                str(sample_documents_jsonl),
                "--output-csv",
                str(output_csv),
                "--limit",
                "2",
                "--dry-run",
                "--concurrency",
                "2",
            ],
        )

        assert result.exit_code == 0
        assert "Vector Classifier Audit Pass" in result.stdout
        assert "Offline Audit Summary" in result.stdout
        assert output_csv.exists()


@respx.mock
def test_load_documents_from_url():
    """Verify loading documents from HTTP/HTTPS Paperless API URL."""
    url = "https://paperless.example.com/api/documents/"
    respx.get(url).respond(
        status_code=200,
        json={
            "count": 2,
            "results": [{"id": 501, "title": "Doc1"}, {"id": 502, "title": "Doc2"}],
        },
    )
    docs = load_documents(url)
    assert len(docs) == 2
    assert docs[0]["id"] == 501
    assert docs[1]["id"] == 502

    # Root list response
    list_url = "https://paperless.example.com/api/list/"
    respx.get(list_url).respond(
        status_code=200,
        json=[{"id": 601, "title": "DirectList"}],
    )
    docs_list = load_documents_from_file(list_url)
    assert len(docs_list) == 1
    assert docs_list[0]["id"] == 601


def test_audit_document_normalization():
    """Verify AuditDocument.from_raw normalizes varied metadata formats."""
    raw = {
        "pk": "42",
        "title": "rechnung.pdf",
        "body": "Rechnungstext",
        "document_type_name": "Rechnung",
        "tags": [{"name": "Fahrzeug"}, "Steuer"],
    }
    doc = AuditDocument.from_raw(raw)
    assert doc.paperless_id == 42
    assert doc.filename == "rechnung.pdf"
    assert doc.content == "Rechnungstext"
    assert doc.current_doc_type == "Rechnung"
    assert "Fahrzeug" in doc.current_tags
    assert "Steuer" in doc.current_tags
    assert doc.is_currently_tax_tagged is True
    assert len(doc.content_sha256) == 64


@pytest.mark.asyncio
async def test_audit_runner_target_collection():
    """Verify target_collection overrides Qdrant collection on the storage manager."""
    mock_engine = AsyncMock(spec=VectorClassifierEngine)
    mock_engine.classify.return_value = VectorClassificationResult(
        predicted_doc_type="Rechnung",
        doc_type_confidence=0.9,
    )
    mock_qdrant = AsyncMock(spec=QdrantStorageManager)
    mock_qdrant.collection = "original_collection"

    runner = AuditRunner(
        engine=mock_engine,
        qdrant_manager=mock_qdrant,
        target_collection="custom_audit_collection",
        index=True,
    )
    assert runner.target_collection == "custom_audit_collection"
    assert mock_qdrant.collection == "custom_audit_collection"
