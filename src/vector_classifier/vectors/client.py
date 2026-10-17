"""Asynchronous Ollama client for generating dense vector embeddings."""

import asyncio
import unicodedata
from typing import Any

import httpx

from vector_classifier.config import settings

TRANSIENT_STATUS_CODES = {429, 502, 503, 504}


class OllamaClientError(Exception):
    """Base exception for Ollama embedding client errors."""


class OllamaConnectionError(OllamaClientError):
    """Raised when connection to Ollama fails after retry attempts."""


class OllamaResponseError(OllamaClientError):
    """Raised when Ollama returns an error status code or invalid response."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def sanitize_text(text: str) -> str:
    """Sanitize input text by stripping control characters and trimming whitespace."""
    if not text:
        return ""
    chars: list[str] = []
    for ch in text:
        if ch in ("\t", "\n", "\r"):
            chars.append(ch)
        elif unicodedata.category(ch) != "Cc" and ord(ch) != 127:
            chars.append(ch)
    return "".join(chars).strip()


class OllamaEmbeddingClient:
    """Asynchronous client for generating embeddings via Ollama.

    Features:
    - Connection pooling and timeout management via httpx.AsyncClient.
    - Automatic retries with exponential backoff for transient errors.
    - Input sanitization and fallback for empty strings.
    - Concurrent header and content embedding generation.
    - Validation of returned embedding dimensions.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        expected_dim: int | None = 1024,
        fallback_text: str = "-",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        raw_url = base_url or settings.ollama_base_url
        self.base_url = raw_url.rstrip("/")
        self.model = model or settings.ollama_embed_model
        self.timeout = timeout if timeout is not None else settings.ollama_timeout
        self.max_retries = max(1, max_retries)
        self.retry_delay = retry_delay
        self.expected_dim = expected_dim
        self.fallback_text = fallback_text
        self._external_client = client is not None
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        """Return or lazily initialize the underlying HTTP client."""
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                limits=limits,
            )
            self._external_client = False
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client if internally managed."""
        if (
            self._client is not None
            and not self._client.is_closed
            and not self._external_client
        ):
            await self._client.aclose()
        self._client = None

    async def __aenter__(self) -> "OllamaEmbeddingClient":
        self._get_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        await self.close()

    async def _post_with_retry(
        self, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Send a POST request with retry logic for transient errors."""
        client = self._get_client()
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await client.post(endpoint, json=payload)
                if (
                    response.status_code in TRANSIENT_STATUS_CODES
                    and attempt < self.max_retries - 1
                ):
                    await asyncio.sleep(self.retry_delay * (2**attempt))
                    continue

                if response.is_error:
                    raise OllamaResponseError(
                        f"Ollama returned HTTP {response.status_code}: {response.text}",
                        status_code=response.status_code,
                    )

                data: Any = response.json()
                if not isinstance(data, dict):
                    raise OllamaResponseError(
                        f"Invalid JSON payload returned from Ollama: {data}"
                    )
                return data

            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exception = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
                    continue
                raise OllamaConnectionError(
                    f"Failed to connect to Ollama at {self.base_url} after {self.max_retries} attempts: {exc}"
                ) from exc

        if last_exception:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama: {last_exception}"
            ) from last_exception
        raise OllamaResponseError("Failed to obtain response from Ollama")

    async def get_embedding(self, text: str) -> list[float]:
        """Generate a dense vector embedding for the given text.

        Sanitizes input by stripping control characters and trimming whitespace.
        If text is empty or consists purely of whitespace/control characters,
        the configured fallback token is embedded instead.
        """
        cleaned = sanitize_text(text)
        if len(cleaned) > 8000:
            cleaned = cleaned[:8000].strip()
        prompt = cleaned if cleaned else self.fallback_text

        payload = {
            "model": self.model,
            "prompt": prompt,
        }

        try:
            data = await self._post_with_retry("/api/embeddings", payload)
        except OllamaResponseError as err:
            if "exceeds the context length" in str(err) and len(prompt) > 1000:
                payload["prompt"] = prompt[: len(prompt) // 2].strip()
                data = await self._post_with_retry("/api/embeddings", payload)
            else:
                raise

        if "embedding" not in data:
            raise OllamaResponseError(
                f"Missing 'embedding' key in Ollama response: {data}"
            )

        raw_embedding = data["embedding"]
        if not isinstance(raw_embedding, list) or len(raw_embedding) == 0:
            raise OllamaResponseError(
                "Ollama returned empty or non-list embedding vector"
            )

        try:
            vector = [float(x) for x in raw_embedding]
        except (ValueError, TypeError) as exc:
            raise OllamaResponseError(
                f"Embedding vector contains non-float values: {exc}"
            ) from exc

        if self.expected_dim is not None and len(vector) != self.expected_dim:
            raise OllamaResponseError(
                f"Expected embedding dimension {self.expected_dim}, got {len(vector)}"
            )

        return vector

    async def get_document_embeddings(
        self, header_text: str, content_text: str
    ) -> tuple[list[float], list[float]]:
        """Concurrently generate embeddings for header and content text sections."""
        header_vector, content_vector = await asyncio.gather(
            self.get_embedding(header_text),
            self.get_embedding(content_text),
        )
        return header_vector, content_vector

    async def health_check(self, check_model: bool = False) -> bool:
        """Check if Ollama service is reachable and responsive.

        Args:
            check_model: If True, also checks that self.model is present in /api/tags.
        """
        try:
            client = self._get_client()
            if check_model:
                resp = await client.get("/api/tags", timeout=5.0)
                if resp.status_code != 200:
                    return False
                data: Any = resp.json()
                models = [
                    m.get("name", "")
                    for m in data.get("models", [])
                    if isinstance(m, dict)
                ]
                return any(self.model in m for m in models)

            resp = await client.get("/api/version", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
