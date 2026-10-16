"""Pytest fixtures and configuration."""

import pytest
from httpx import ASGITransport, AsyncClient

from vector_classifier.main import app


@pytest.fixture
async def async_client():
    """Async client fixture for testing FastAPI endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
