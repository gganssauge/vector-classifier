"""Qdrant storage manager and schema management."""

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from vector_classifier.config import settings

logger = logging.getLogger(__name__)

PAYLOAD_INDEXES: list[tuple[str, models.PayloadSchemaType]] = [
    ("sha256", models.PayloadSchemaType.KEYWORD),
    ("paperless_id", models.PayloadSchemaType.INTEGER),
    ("doc_type", models.PayloadSchemaType.KEYWORD),
    ("is_tax_relevant", models.PayloadSchemaType.BOOL),
    ("tax_category", models.PayloadSchemaType.KEYWORD),
]


class QdrantStorageError(Exception):
    """Base exception for Qdrant storage operations."""


class QdrantConnectionError(QdrantStorageError):
    """Raised when connection to Qdrant fails."""


def generate_point_id(content_hash: str) -> str:
    """Generate reproducible UUIDv5 from document content SHA-256 hash.

    Args:
        content_hash: SHA-256 hex string of the document content.

    Returns:
        Deterministic UUIDv5 string representation.
    """
    clean_hash = content_hash.strip().lower()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, clean_hash))


def _build_filter(
    criteria: dict[str, Any] | models.Filter | None,
) -> models.Filter | None:
    """Build a Qdrant Filter from a dictionary or return existing Filter."""
    if criteria is None:
        return None
    if isinstance(criteria, models.Filter):
        return criteria

    must_conditions: list[models.FieldCondition] = []
    for key, value in criteria.items():
        if isinstance(value, list):
            condition = models.FieldCondition(
                key=key,
                match=models.MatchAny(any=value),
            )
        else:
            condition = models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value),
            )
        must_conditions.append(condition)

    return models.Filter(must=must_conditions)


class QdrantStorageManager:
    """Asynchronous manager for Qdrant vector storage and schema operations.

    Features:
    - Idempotent collection initialization with named vectors ('header', 'content').
    - Creation of payload indexes for metadata filtering.
    - Idempotent document point upsert.
    - Vector similarity search with payload filtering.
    - Connection health checking and async context manager support.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        """Initialize QdrantStorageManager with configuration or defaults."""
        self.host = host if host is not None else settings.qdrant_host
        self.port = port if port is not None else settings.qdrant_port
        self.collection = (
            collection if collection is not None else settings.qdrant_collection
        )
        self.api_key = api_key if api_key is not None else settings.qdrant_api_key
        self.timeout = timeout if timeout is not None else settings.qdrant_timeout
        self._external_client = client is not None
        self._client = client

    @property
    def client(self) -> AsyncQdrantClient:
        """Return or lazily initialize the AsyncQdrantClient."""
        if self._client is None:
            self._client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key if self.api_key else None,
                timeout=int(self.timeout) if self.timeout is not None else None,
            )
        return self._client

    @client.setter
    def client(self, value: AsyncQdrantClient) -> None:
        """Set the underlying AsyncQdrantClient."""
        self._client = value
        self._external_client = True

    async def ensure_collection(self) -> None:
        """Idempotently ensure collection exists with named vectors and payload indexes."""
        try:
            if await self.client.collection_exists(self.collection):
                logger.info("Collection '%s' already exists", self.collection)
                return

            logger.info("Creating collection '%s' with named vectors", self.collection)
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "header": models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE,
                    ),
                    "content": models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE,
                    ),
                },
            )

            for field_name, schema_type in PAYLOAD_INDEXES:
                logger.info(
                    "Creating payload index for '%s' (%s)", field_name, schema_type
                )
                await self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema_type,
                )
        except Exception as e:
            logger.error("Failed to ensure collection '%s': %s", self.collection, e)
            raise QdrantStorageError(
                f"Failed to ensure collection '{self.collection}': {e}"
            ) from e

    async def recreate_collection(self) -> None:
        """Drop collection if it exists and recreate with schema and payload indexes."""
        try:
            if await self.client.collection_exists(self.collection):
                logger.info("Dropping existing collection '%s'", self.collection)
                await self.client.delete_collection(self.collection)
            await self.ensure_collection()
        except Exception as e:
            logger.error("Failed to recreate collection '%s': %s", self.collection, e)
            raise QdrantStorageError(
                f"Failed to recreate collection '{self.collection}': {e}"
            ) from e

    async def upsert_documents_batch(
        self,
        points: list[models.PointStruct],
        wait: bool = False,
    ) -> None:
        """Upsert a batch of document points into Qdrant.

        Args:
            points: List of PointStruct instances to upsert.
            wait: Whether to wait for changes to be committed to disk.
        """
        if not points:
            return
        try:
            await self.client.upsert(
                collection_name=self.collection,
                points=points,
                wait=wait,
            )
        except Exception as e:
            logger.error("Failed to batch upsert %d points: %s", len(points), e)
            raise QdrantStorageError(
                f"Failed to batch upsert {len(points)} points: {e}"
            ) from e

    async def upsert_document(
        self,
        point_id: str,
        header_vector: list[float],
        content_vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Upsert a single document into Qdrant using named vectors and payload.

        Args:
            point_id: Deterministic point identifier.
            header_vector: 1024-dimensional header embedding vector.
            content_vector: 1024-dimensional content embedding vector.
            payload: Metadata payload for the document point.
        """
        try:
            point = models.PointStruct(
                id=point_id,
                vector={
                    "header": header_vector,
                    "content": content_vector,
                },
                payload=payload,
            )
            await self.upsert_documents_batch(points=[point], wait=False)
        except Exception as e:
            logger.error("Failed to upsert point '%s': %s", point_id, e)
            raise QdrantStorageError(
                f"Failed to upsert document '{point_id}': {e}"
            ) from e

    async def search_by_vector(
        self,
        vector_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float | None = None,
        filter_criteria: dict[str, Any] | models.Filter | None = None,
    ) -> list[Any]:
        """Search for similar points using a named vector and optional filter criteria.

        Args:
            vector_name: Name of vector ('header' or 'content').
            vector: Query embedding vector.
            limit: Maximum number of points to retrieve.
            score_threshold: Minimum cosine similarity score threshold.
            filter_criteria: Optional dictionary of filter criteria or Filter object.

        Returns:
            List of scored points.
        """
        query_filter = _build_filter(filter_criteria)
        try:
            if hasattr(self.client, "query_points"):
                response = await self.client.query_points(
                    collection_name=self.collection,
                    query=vector,
                    using=vector_name,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                )
                return (
                    response.points if hasattr(response, "points") else response  # type: ignore[return-value]
                )
            if hasattr(self.client, "search"):
                return await self.client.search(  # type: ignore[attr-defined]
                    collection_name=self.collection,
                    query_vector=(vector_name, vector),
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                )
            raise QdrantStorageError(
                "Underlying client does not support query_points or search"
            )
        except QdrantStorageError:
            raise
        except Exception as e:
            logger.error("Vector search failed for '%s': %s", vector_name, e)
            raise QdrantStorageError(
                f"Vector search failed for '{vector_name}': {e}"
            ) from e

    async def health_check(self) -> bool:
        """Check if Qdrant service is reachable and responsive."""
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close underlying client connection if internally managed."""
        if self._client is not None and not self._external_client:
            await self._client.close()
        self._client = None

    async def __aenter__(self) -> "QdrantStorageManager":
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit async context manager and close client."""
        await self.close()
