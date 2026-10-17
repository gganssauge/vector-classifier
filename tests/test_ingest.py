"""Unit tests for Paperless document ingestion pipeline."""

from unittest.mock import AsyncMock

import pytest
import respx
from httpx import Response
from typer.testing import CliRunner

from vector_classifier.cli import app
from vector_classifier.ingest import PaperlessIngestRunner
from vector_classifier.models.schemas import VectorClassificationResult
from vector_classifier.vectors.classifier import VectorClassifierEngine
from vector_classifier.vectors.qdrant import QdrantStorageManager


@pytest.fixture
def mock_engine():
    """Mock VectorClassifierEngine returning canned classification results."""
    engine = AsyncMock(spec=VectorClassifierEngine)
    engine.initialize_anchor_embeddings = AsyncMock()
    engine.classify.return_value = VectorClassificationResult(
        predicted_doc_type="Rechnung",
        doc_type_confidence=0.92,
        predicted_tags=["Haushalt"],
        tag_scores={"Haushalt": 0.88},
        is_tax_relevant=True,
        tax_confidence=0.85,
        tax_category="werbungskosten",
        header_vector=[0.1] * 8,
        content_vector=[0.2] * 8,
    )
    return engine


@pytest.fixture
def mock_qdrant():
    """Mock QdrantStorageManager."""
    qdrant = AsyncMock(spec=QdrantStorageManager)
    qdrant.collection = "paperless_documents"
    qdrant.ensure_collection = AsyncMock()
    qdrant.recreate_collection = AsyncMock()
    qdrant.upsert_documents_batch = AsyncMock()
    return qdrant


@pytest.mark.asyncio
async def test_paperless_ingest_runner_flow(mock_engine, mock_qdrant):
    """Verify PaperlessIngestRunner successfully streams, normalizes, vectorizes, and indexes."""
    base_url = "https://mock.paperless.test"

    with respx.mock(assert_all_called=False) as respx_mock:
        # Mock tags endpoint
        respx_mock.get(f"{base_url}/api/tags/?page_size=1000").mock(
            return_value=Response(
                200,
                json={
                    "results": [
                        {"id": 1, "name": "Steuer"},
                        {"id": 2, "name": "Rechnung"},
                    ]
                },
            )
        )

        # Mock document_types endpoint
        respx_mock.get(f"{base_url}/api/document_types/?page_size=1000").mock(
            return_value=Response(
                200,
                json={"results": [{"id": 10, "name": "Rechnung"}]},
            )
        )

        # Mock probe count endpoint
        respx_mock.get(f"{base_url}/api/documents/?page_size=1").mock(
            return_value=Response(200, json={"count": 2})
        )

        # Mock paginated documents
        respx_mock.get(f"{base_url}/api/documents/?page=1&page_size=2").mock(
            return_value=Response(
                200,
                json={
                    "count": 2,
                    "next": None,
                    "results": [
                        {
                            "id": 101,
                            "title": "Telekom Rechnung",
                            "document_type": 10,
                            "tags": [1, 2],
                            "content": "Rechnung über Internetzugang März 2026",
                            "original_file_name": "telekom.pdf",
                        },
                        {
                            "id": 102,
                            "title": "Empty Scan",
                            "document_type": None,
                            "tags": [],
                            "content": "",
                            "original_file_name": "empty.pdf",
                        },
                    ],
                },
            )
        )

        runner = PaperlessIngestRunner(
            paperless_url=base_url,
            paperless_token="dummy-token",
            engine=mock_engine,
            qdrant_manager=mock_qdrant,
            concurrency=2,
            batch_size=10,
            recreate_collection=True,
        )

        summary = await runner.run()

        mock_qdrant.recreate_collection.assert_awaited_once()
        mock_engine.initialize_anchor_embeddings.assert_awaited_once()

        assert summary.total_scanned == 2
        assert summary.total_indexed == 1
        assert summary.total_skipped_empty == 1
        assert summary.tax_relevant_count == 1

        # Verify batch upsert was called with valid point
        mock_qdrant.upsert_documents_batch.assert_awaited_once()
        points_arg = mock_qdrant.upsert_documents_batch.call_args[0][0]
        assert len(points_arg) == 1
        assert points_arg[0].payload["paperless_id"] == 101
        assert points_arg[0].payload["doc_type_original"] == "Rechnung"
        assert "Steuer" in points_arg[0].payload["tags_original"]
        assert points_arg[0].payload["is_tax_relevant"] is True


def test_cli_ingest_help():
    """Verify CLI ingest command help displays properly."""
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "Ingest documents from Paperless-ngx" in result.stdout
    assert "--recreate" in result.stdout
    assert "--concurrency" in result.stdout
    assert "--batch-size" in result.stdout
