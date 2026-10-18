"""Unit tests for Qdrant storage manager and collection schema."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models

from vector_classifier.config import settings
from vector_classifier.vectors.qdrant import (
    PAYLOAD_INDEXES,
    QdrantStorageError,
    QdrantStorageManager,
    _build_filter,
    generate_point_id,
)


def test_generate_point_id_reproducibility():
    """Verify that generate_point_id produces deterministic and valid UUIDv5 values."""
    hash_1 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    id_1 = generate_point_id(hash_1)
    id_2 = generate_point_id(hash_1)
    assert id_1 == id_2

    # Valid UUID format
    parsed = uuid.UUID(id_1)
    assert parsed.version == 5

    # Case insensitivity & trimming
    id_upper = generate_point_id(f"  {hash_1.upper()}  ")
    assert id_1 == id_upper


def test_generate_point_id_different_hashes():
    """Verify that different content hashes result in distinct point IDs."""
    hash_a = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    hash_b = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
    assert generate_point_id(hash_a) != generate_point_id(hash_b)


def test_build_filter_none():
    """Verify _build_filter returns None when criteria is None."""
    assert _build_filter(None) is None


def test_build_filter_existing_filter():
    """Verify _build_filter returns an existing Filter instance untouched."""
    existing = models.Filter(must=[])
    assert _build_filter(existing) is existing


def test_build_filter_dict():
    """Verify _build_filter constructs FieldCondition matches properly."""
    criteria = {
        "doc_type": "Rechnung",
        "paperless_id": 123,
        "is_tax_relevant": True,
        "tags": ["Finanzen", "Versicherung"],
    }
    f = _build_filter(criteria)
    assert isinstance(f, models.Filter)
    assert isinstance(f.must, list)
    assert len(f.must) == 4

    keys = {c.key: c.match for c in f.must if isinstance(c, models.FieldCondition)}
    assert keys["doc_type"] == models.MatchValue(value="Rechnung")
    assert keys["paperless_id"] == models.MatchValue(value=123)
    assert keys["is_tax_relevant"] == models.MatchValue(value=True)
    assert keys["tags"] == models.MatchAny(any=["Finanzen", "Versicherung"])


def test_manager_defaults():
    """Verify default configurations align with application settings."""
    manager = QdrantStorageManager()
    assert manager.host == settings.qdrant_host
    assert manager.port == settings.qdrant_port
    assert manager.collection == settings.qdrant_collection
    assert manager.api_key == settings.qdrant_api_key


def test_manager_custom_config():
    """Verify custom configurations are correctly stored."""
    manager = QdrantStorageManager(
        host="custom-qdrant",
        port=6334,
        collection="custom_collection",
        api_key="secret-key",
    )
    assert manager.host == "custom-qdrant"
    assert manager.port == 6334
    assert manager.collection == "custom_collection"
    assert manager.api_key == "secret-key"


@pytest.mark.asyncio
async def test_ensure_collection_creates_when_not_exists():
    """Verify collection and payload indexes are created when collection does not exist."""
    mock_client = AsyncMock()
    mock_client.collection_exists.return_value = False
    mock_client.create_collection.return_value = True
    mock_client.create_payload_index.return_value = MagicMock()

    manager = QdrantStorageManager(client=mock_client, collection="test_docs")
    await manager.ensure_collection()

    mock_client.collection_exists.assert_awaited_once_with("test_docs")
    mock_client.create_collection.assert_awaited_once()

    call_kwargs = mock_client.create_collection.call_args.kwargs
    assert call_kwargs["collection_name"] == "test_docs"
    vectors_config = call_kwargs["vectors_config"]
    assert "header" in vectors_config
    assert "content" in vectors_config
    assert vectors_config["header"].size == 1024
    assert vectors_config["header"].distance == models.Distance.COSINE
    assert vectors_config["content"].size == 1024
    assert vectors_config["content"].distance == models.Distance.COSINE

    assert mock_client.create_payload_index.await_count == len(PAYLOAD_INDEXES)
    created_fields = [
        call.kwargs["field_name"]
        for call in mock_client.create_payload_index.call_args_list
    ]
    assert created_fields == [
        "sha256",
        "paperless_id",
        "doc_type",
        "is_tax_relevant",
        "is_tax_curated",
        "tax_category",
    ]


@pytest.mark.asyncio
async def test_ensure_collection_skips_when_exists():
    """Verify ensure_collection is a no-op if collection already exists."""
    mock_client = AsyncMock()
    mock_client.collection_exists.return_value = True

    manager = QdrantStorageManager(client=mock_client, collection="test_docs")
    await manager.ensure_collection()

    mock_client.collection_exists.assert_awaited_once_with("test_docs")
    mock_client.create_collection.assert_not_awaited()
    mock_client.create_payload_index.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_collection_raises_on_error():
    """Verify ensure_collection wraps exceptions into QdrantStorageError."""
    mock_client = AsyncMock()
    mock_client.collection_exists.side_effect = RuntimeError("Connection refused")

    manager = QdrantStorageManager(client=mock_client)
    with pytest.raises(QdrantStorageError, match="Failed to ensure collection"):
        await manager.ensure_collection()


@pytest.mark.asyncio
async def test_upsert_document_success():
    """Verify single document upsert creates the right PointStruct with named vectors."""
    mock_client = AsyncMock()
    mock_client.upsert.return_value = MagicMock()

    manager = QdrantStorageManager(client=mock_client, collection="test_docs")
    header_vec = [0.1] * 1024
    content_vec = [0.2] * 1024
    payload = {
        "paperless_id": 42,
        "sha256": "abcdef123456",
        "doc_type": "Rechnung",
        "is_tax_relevant": True,
    }

    await manager.upsert_document(
        point_id="test-point-id",
        header_vector=header_vec,
        content_vector=content_vec,
        payload=payload,
    )

    mock_client.upsert.assert_awaited_once()
    kwargs = mock_client.upsert.call_args.kwargs
    assert kwargs["collection_name"] == "test_docs"
    points = kwargs["points"]
    assert len(points) == 1
    point = points[0]
    assert point.id == "test-point-id"
    assert point.vector["header"] == header_vec
    assert point.vector["content"] == content_vec
    assert point.payload == payload


@pytest.mark.asyncio
async def test_upsert_document_error():
    """Verify upsert_document wraps failures in QdrantStorageError."""
    mock_client = AsyncMock()
    mock_client.upsert.side_effect = RuntimeError("Upsert failed")

    manager = QdrantStorageManager(client=mock_client)
    with pytest.raises(QdrantStorageError, match="Failed to upsert document"):
        await manager.upsert_document(
            point_id="test-id",
            header_vector=[0.0],
            content_vector=[0.0],
            payload={},
        )


@pytest.mark.asyncio
async def test_search_by_vector_query_points():
    """Verify search_by_vector uses query_points when available."""
    mock_client = AsyncMock()
    scored_point = MagicMock()
    scored_point.id = "doc-1"
    scored_point.score = 0.95
    query_response = MagicMock()
    query_response.points = [scored_point]
    mock_client.query_points.return_value = query_response

    manager = QdrantStorageManager(client=mock_client, collection="test_docs")
    query_vec = [0.1] * 1024
    results = await manager.search_by_vector(
        vector_name="content",
        vector=query_vec,
        limit=3,
        score_threshold=0.8,
        filter_criteria={"is_tax_relevant": True},
    )

    assert results == [scored_point]
    mock_client.query_points.assert_awaited_once()
    kwargs = mock_client.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "test_docs"
    assert kwargs["query"] == query_vec
    assert kwargs["using"] == "content"
    assert kwargs["limit"] == 3
    assert kwargs["score_threshold"] == 0.8
    assert isinstance(kwargs["query_filter"], models.Filter)


@pytest.mark.asyncio
async def test_search_by_vector_fallback_search():
    """Verify search_by_vector falls back to search when query_points is missing."""
    mock_client = MagicMock(spec=["search"])
    mock_client.search = AsyncMock(return_value=["point-a"])

    manager = QdrantStorageManager(client=mock_client, collection="test_docs")
    query_vec = [0.1] * 1024
    results = await manager.search_by_vector(
        vector_name="header",
        vector=query_vec,
        limit=5,
    )

    assert results == ["point-a"]
    mock_client.search.assert_awaited_once()
    kwargs = mock_client.search.call_args.kwargs
    assert kwargs["collection_name"] == "test_docs"
    assert kwargs["query_vector"] == ("header", query_vec)
    assert kwargs["limit"] == 5
    assert kwargs["query_filter"] is None


@pytest.mark.asyncio
async def test_search_by_vector_error():
    """Verify search_by_vector wraps errors in QdrantStorageError."""
    mock_client = AsyncMock()
    mock_client.query_points.side_effect = RuntimeError("Search timeout")

    manager = QdrantStorageManager(client=mock_client)
    with pytest.raises(QdrantStorageError, match="Vector search failed"):
        await manager.search_by_vector(
            vector_name="content",
            vector=[0.1] * 1024,
        )


@pytest.mark.asyncio
async def test_health_check_success():
    """Verify health_check returns True when get_collections succeeds."""
    mock_client = AsyncMock()
    mock_client.get_collections.return_value = MagicMock()

    manager = QdrantStorageManager(client=mock_client)
    assert await manager.health_check() is True
    mock_client.get_collections.assert_awaited_once()


@pytest.mark.asyncio
async def test_health_check_failure():
    """Verify health_check returns False when get_collections raises an exception."""
    mock_client = AsyncMock()
    mock_client.get_collections.side_effect = ConnectionRefusedError("Unreachable")

    manager = QdrantStorageManager(client=mock_client)
    assert await manager.health_check() is False


@pytest.mark.asyncio
async def test_context_manager_and_close():
    """Verify async context manager calls close on internally created client."""
    mock_client = AsyncMock()
    manager = QdrantStorageManager()
    manager._client = mock_client
    manager._external_client = False

    async with manager as m:
        assert m is manager

    mock_client.close.assert_awaited_once()
    assert manager._client is None


@pytest.mark.asyncio
async def test_recreate_collection_success():
    """Verify recreate_collection deletes existing collection and calls ensure_collection."""
    mock_client = AsyncMock()
    mock_client.collection_exists.side_effect = [True, False]
    manager = QdrantStorageManager(client=mock_client)

    await manager.recreate_collection()

    mock_client.delete_collection.assert_awaited_once_with(manager.collection)
    mock_client.create_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_recreate_collection_error():
    """Verify recreate_collection raises QdrantStorageError on failure."""
    mock_client = AsyncMock()
    mock_client.collection_exists.side_effect = RuntimeError("Qdrant error")
    manager = QdrantStorageManager(client=mock_client)

    with pytest.raises(QdrantStorageError, match="Failed to recreate collection"):
        await manager.recreate_collection()


@pytest.mark.asyncio
async def test_upsert_documents_batch_success():
    """Verify upsert_documents_batch calls client.upsert with points list."""
    mock_client = AsyncMock()
    manager = QdrantStorageManager(client=mock_client)
    points = [
        models.PointStruct(id="1", vector={"content": [0.1]}, payload={"test": 1}),
        models.PointStruct(id="2", vector={"content": [0.2]}, payload={"test": 2}),
    ]

    await manager.upsert_documents_batch(points)
    mock_client.upsert.assert_awaited_once_with(
        collection_name=manager.collection,
        points=points,
        wait=False,
    )


@pytest.mark.asyncio
async def test_upsert_documents_batch_empty():
    """Verify upsert_documents_batch does nothing if points list is empty."""
    mock_client = AsyncMock()
    manager = QdrantStorageManager(client=mock_client)

    await manager.upsert_documents_batch([])
    mock_client.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_upsert_documents_batch_error():
    """Verify upsert_documents_batch raises QdrantStorageError on failure."""
    mock_client = AsyncMock()
    mock_client.upsert.side_effect = RuntimeError("Upsert error")
    manager = QdrantStorageManager(client=mock_client)
    points = [models.PointStruct(id="1", vector={"content": [0.1]}, payload={})]

    with pytest.raises(QdrantStorageError, match="Failed to batch upsert"):
        await manager.upsert_documents_batch(points)
