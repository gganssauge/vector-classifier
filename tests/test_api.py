"""Tests for FastAPI endpoints."""

from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from vector_classifier import __version__
from vector_classifier.main import set_engine
from vector_classifier.models.schemas import VectorClassificationResult
from vector_classifier.vectors.classifier import VectorClassifierEngine


@pytest.fixture(autouse=True)
def cleanup_engine():
    """Ensure engine state is cleaned up after each test."""
    yield
    set_engine(None)


@pytest.mark.asyncio
async def test_health_endpoint_healthy(async_client: AsyncClient):
    """Test health check reflecting connected services."""
    mock_ollama = AsyncMock()
    mock_ollama.health_check.return_value = True

    mock_qdrant = AsyncMock()
    mock_qdrant.health_check.return_value = True

    mock_engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=mock_qdrant,
    )
    set_engine(mock_engine)

    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == __version__
    assert data["ollama_connected"] is True
    assert data["qdrant_connected"] is True


@pytest.mark.asyncio
async def test_health_endpoint_disconnected(async_client: AsyncClient):
    """Test health check reflecting disconnected services."""
    mock_ollama = AsyncMock()
    mock_ollama.health_check.return_value = False

    mock_qdrant = AsyncMock()
    mock_qdrant.health_check.return_value = False

    mock_engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=mock_qdrant,
    )
    set_engine(mock_engine)

    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ollama_connected"] is False
    assert data["qdrant_connected"] is False


@pytest.mark.asyncio
async def test_classify_endpoint(async_client: AsyncClient):
    """Test classify endpoint invoking VectorClassifierEngine."""
    mock_engine = AsyncMock(spec=VectorClassifierEngine)
    mock_engine.classify.return_value = VectorClassificationResult(
        predicted_doc_type="Rechnung",
        doc_type_confidence=0.92,
        predicted_tags=["IT", "Telekommunikation"],
        tag_scores={"IT": 0.88, "Telekommunikation": 0.82},
        is_tax_relevant=True,
        tax_confidence=0.85,
        tax_category="steuer:werbungskosten",
        tax_review_required=False,
        matched_anchors=[
            {"name": "Rechnung", "category": "doc_type", "similarity": 0.92}
        ],
    )
    set_engine(mock_engine)

    payload = {
        "content": "Rechnung über IT-Dienstleistungen und Cloud Hosting",
        "filename": "rechnung_2026.pdf",
        "sha256": "abcdef1234567890",
    }
    response = await async_client.post("/api/v1/classify", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_doc_type"] == "Rechnung"
    assert data["doc_type_confidence"] == 0.92
    assert data["predicted_tags"] == ["IT", "Telekommunikation"]
    assert data["is_tax_relevant"] is True
    assert data["tax_confidence"] == 0.85
    assert data["tax_category"] == "steuer:werbungskosten"
    assert data["tax_review_required"] is False
    assert len(data["matched_anchors"]) == 1
    mock_engine.classify.assert_awaited_once_with(
        content=payload["content"],
        filename=payload["filename"],
        sha256=payload["sha256"],
        header_content=None,
    )


@pytest.mark.asyncio
async def test_lifespan():
    """Test FastAPI lifespan startup and shutdown cycle."""
    from vector_classifier.main import app, lifespan

    mock_engine = AsyncMock(spec=VectorClassifierEngine)
    set_engine(mock_engine)

    async with lifespan(app):
        mock_engine.initialize_anchor_embeddings.assert_awaited_once()

    mock_engine.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_initialization_failure():
    """Lifespan should defer anchor initialization if an error occurs."""
    from vector_classifier.main import app, lifespan

    mock_engine = AsyncMock(spec=VectorClassifierEngine)
    mock_engine.initialize_anchor_embeddings.side_effect = RuntimeError("Ollama down")
    set_engine(mock_engine)

    async with lifespan(app):
        pass

    mock_engine.close.assert_awaited_once()
