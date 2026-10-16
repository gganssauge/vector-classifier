"""Tests for FastAPI endpoints."""

import pytest
from httpx import AsyncClient

from vector_classifier import __version__


@pytest.mark.asyncio
async def test_health_endpoint(async_client: AsyncClient):
    """Test health check response."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == __version__


@pytest.mark.asyncio
async def test_classify_endpoint_stub(async_client: AsyncClient):
    """Test classify stub response."""
    payload = {
        "content": "Rechnung über IT-Dienstleistungen",
        "filename": "test.pdf",
        "sha256": "dummyhash",
    }
    response = await async_client.post("/api/v1/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_doc_type" in data
    assert "is_tax_relevant" in data
