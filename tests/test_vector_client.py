"""Unit tests for the Ollama embedding client."""

import json

import httpx
import pytest

from vector_classifier.config import settings
from vector_classifier.vectors.client import (
    OllamaConnectionError,
    OllamaEmbeddingClient,
    OllamaResponseError,
    sanitize_text,
)


def test_sanitize_text():
    """Verify control character removal, whitespace trimming, and preservation of newlines."""
    assert sanitize_text("") == ""
    assert sanitize_text("   ") == ""
    assert sanitize_text(" \x00\x08\x1f ") == ""
    assert sanitize_text("Hello\x00 World\x1b") == "Hello World"
    assert sanitize_text("  Zeile 1\nZeile 2\tTab  ") == "Zeile 1\nZeile 2\tTab"


def test_client_defaults():
    """Verify default configurations match application settings."""
    client = OllamaEmbeddingClient()
    assert client.base_url == settings.ollama_base_url.rstrip("/")
    assert client.model == settings.ollama_embed_model
    assert client.timeout == settings.ollama_timeout
    assert client.max_retries == 3
    assert client.expected_dim == 1024


def test_client_custom_init():
    """Verify custom configuration options are correctly applied."""
    client = OllamaEmbeddingClient(
        base_url="http://custom-ollama:11434/",
        model="custom-model",
        timeout=15.0,
        max_retries=5,
        expected_dim=512,
        fallback_text="empty",
    )
    assert client.base_url == "http://custom-ollama:11434"
    assert client.model == "custom-model"
    assert client.timeout == 15.0
    assert client.max_retries == 5
    assert client.expected_dim == 512
    assert client.fallback_text == "empty"


async def test_get_embedding_success(respx_mock):
    """Verify get_embedding returns a valid 1024-dimensional float vector."""
    base_url = "http://test-ollama:11434"
    mock_vector = [0.001 * (i % 50) for i in range(1024)]
    route = respx_mock.post(f"{base_url}/api/embeddings").respond(
        status_code=200,
        json={"embedding": mock_vector},
    )

    async with OllamaEmbeddingClient(base_url=base_url) as client:
        vector = await client.get_embedding("Rechnung März 2026")

    assert len(vector) == 1024
    assert vector == mock_vector
    assert route.called
    payload = json.loads(route.calls.last.request.content)
    assert payload["model"] == settings.ollama_embed_model
    assert payload["prompt"] == "Rechnung März 2026"


async def test_get_document_embeddings_concurrent(respx_mock):
    """Verify get_document_embeddings issues parallel calls for header and content."""
    base_url = "http://test-ollama:11434"
    header_vector = [0.1] * 1024
    content_vector = [0.2] * 1024

    def side_effect(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        if "Briefkopf" in data.get("prompt", ""):
            return httpx.Response(200, json={"embedding": header_vector})
        return httpx.Response(200, json={"embedding": content_vector})

    route = respx_mock.post(f"{base_url}/api/embeddings").mock(side_effect=side_effect)

    async with OllamaEmbeddingClient(base_url=base_url) as client:
        h_vec, c_vec = await client.get_document_embeddings(
            header_text="Briefkopf Stadtwerke",
            content_text="Stromabrechnung Jahresübersicht",
        )

    assert h_vec == header_vector
    assert c_vec == content_vector
    assert route.call_count == 2


async def test_retry_on_transient_error(respx_mock):
    """Verify retry logic recovers from transient HTTP 503 errors."""
    base_url = "http://test-ollama:11434"
    mock_vector = [0.3] * 1024

    route = respx_mock.post(f"{base_url}/api/embeddings")
    route.side_effect = [
        httpx.Response(503, text="Service Temporarily Unavailable"),
        httpx.Response(200, json={"embedding": mock_vector}),
    ]

    async with OllamaEmbeddingClient(
        base_url=base_url,
        max_retries=3,
        retry_delay=0.001,
    ) as client:
        result = await client.get_embedding("Test Text")

    assert result == mock_vector
    assert route.call_count == 2


async def test_persistent_http_error(respx_mock):
    """Verify persistent HTTP 500 error raises OllamaResponseError."""
    base_url = "http://test-ollama:11434"
    respx_mock.post(f"{base_url}/api/embeddings").respond(
        status_code=500,
        text="Internal Server Error",
    )

    async with OllamaEmbeddingClient(
        base_url=base_url,
        max_retries=2,
        retry_delay=0.001,
    ) as client:
        with pytest.raises(OllamaResponseError) as exc_info:
            await client.get_embedding("Test Text")
        assert exc_info.value.status_code == 500


async def test_persistent_connection_error(respx_mock):
    """Verify persistent network connectivity failure raises OllamaConnectionError."""
    base_url = "http://test-ollama:11434"
    respx_mock.post(f"{base_url}/api/embeddings").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    async with OllamaEmbeddingClient(
        base_url=base_url,
        max_retries=2,
        retry_delay=0.001,
    ) as client:
        with pytest.raises(OllamaConnectionError):
            await client.get_embedding("Test Text")


async def test_health_check_reachable(respx_mock):
    """Verify health_check returns True on HTTP 200 response."""
    base_url = "http://test-ollama:11434"
    respx_mock.get(f"{base_url}/api/version").respond(
        status_code=200,
        json={"version": "0.1.30"},
    )

    async with OllamaEmbeddingClient(base_url=base_url) as client:
        assert await client.health_check() is True


async def test_health_check_unreachable_or_error(respx_mock):
    """Verify health_check returns False on HTTP 500 or network error."""
    base_url = "http://test-ollama:11434"
    version_route = respx_mock.get(f"{base_url}/api/version")

    version_route.respond(status_code=500, text="Crash")
    async with OllamaEmbeddingClient(base_url=base_url) as client:
        assert await client.health_check() is False

    version_route.mock(side_effect=httpx.ConnectError("Unreachable"))
    async with OllamaEmbeddingClient(base_url=base_url) as client:
        assert await client.health_check() is False


async def test_health_check_model_presence(respx_mock):
    """Verify health_check with check_model=True verifies model in /api/tags."""
    base_url = "http://test-ollama:11434"
    tags_route = respx_mock.get(f"{base_url}/api/tags")

    tags_route.respond(
        status_code=200,
        json={"models": [{"name": "bge-m3:latest"}, {"name": "llama3:latest"}]},
    )
    async with OllamaEmbeddingClient(base_url=base_url, model="bge-m3") as client:
        assert await client.health_check(check_model=True) is True

    tags_route.respond(
        status_code=200,
        json={"models": [{"name": "nomic-embed-text:latest"}]},
    )
    async with OllamaEmbeddingClient(base_url=base_url, model="bge-m3") as client:
        assert await client.health_check(check_model=True) is False


async def test_sanitization_and_empty_fallback(respx_mock):
    """Verify input text is sanitized and empty text receives fallback token."""
    base_url = "http://test-ollama:11434"
    mock_vector = [0.1] * 1024
    route = respx_mock.post(f"{base_url}/api/embeddings").respond(
        status_code=200,
        json={"embedding": mock_vector},
    )

    async with OllamaEmbeddingClient(base_url=base_url, fallback_text="-") as client:
        # Sanitization removes control chars but keeps newlines and tabs
        await client.get_embedding("  \x00\x07Rechnung\tBetrag\x1f\nDetails  ")
        req1 = json.loads(route.calls[0].request.content)
        assert req1["prompt"] == "Rechnung\tBetrag\nDetails"

        # Empty or whitespace-only input uses fallback
        await client.get_embedding("   \t\n\x00   ")
        req2 = json.loads(route.calls[1].request.content)
        assert req2["prompt"] == "-"


async def test_dimension_mismatch_error(respx_mock):
    """Verify unexpected vector dimensions trigger OllamaResponseError."""
    base_url = "http://test-ollama:11434"
    respx_mock.post(f"{base_url}/api/embeddings").respond(
        status_code=200,
        json={"embedding": [0.1] * 512},
    )

    async with OllamaEmbeddingClient(base_url=base_url, expected_dim=1024) as client:
        with pytest.raises(OllamaResponseError) as exc_info:
            await client.get_embedding("Valid text")
        assert "dimension" in str(exc_info.value).lower()


async def test_invalid_response_payload(respx_mock):
    """Verify missing or malformed embedding vector triggers OllamaResponseError."""
    base_url = "http://test-ollama:11434"
    route = respx_mock.post(f"{base_url}/api/embeddings")

    # Missing embedding key
    route.respond(status_code=200, json={"result": []})
    async with OllamaEmbeddingClient(base_url=base_url) as client:
        with pytest.raises(OllamaResponseError):
            await client.get_embedding("Sample text")

    # Non-float values in embedding
    route.respond(status_code=200, json={"embedding": ["not", "numbers"]})
    async with OllamaEmbeddingClient(base_url=base_url, expected_dim=None) as client:
        with pytest.raises(OllamaResponseError):
            await client.get_embedding("Sample text")

    # Empty embedding list
    route.respond(status_code=200, json={"embedding": []})
    async with OllamaEmbeddingClient(base_url=base_url) as client:
        with pytest.raises(OllamaResponseError):
            await client.get_embedding("Sample text")

    # Non-dict JSON payload
    route.respond(status_code=200, json=["unexpected", "array"])
    async with OllamaEmbeddingClient(base_url=base_url) as client:
        with pytest.raises(OllamaResponseError):
            await client.get_embedding("Sample text")


async def test_health_check_tags_error(respx_mock):
    """Verify health_check with check_model=True handles non-200 status."""
    base_url = "http://test-ollama:11434"
    respx_mock.get(f"{base_url}/api/tags").respond(
        status_code=500,
        text="Internal Server Error",
    )
    async with OllamaEmbeddingClient(base_url=base_url, model="bge-m3") as client:
        assert await client.health_check(check_model=True) is False
