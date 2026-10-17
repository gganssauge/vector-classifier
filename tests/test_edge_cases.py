"""Comprehensive edge case and regression test suite for vector classification.

Tests cover:
- Empty OCR texts, missing titles, whitespace-only texts.
- Very long texts (>8,192 tokens / multi-page documents) and Header vs Content splitting.
- Precise boundary value tests for the three-zone tax logic (0.69, 0.70, 0.81, 0.82, sub-cent boundaries).
- Qdrant Storage Manager edge cases (unreachable host, malformed filters, existing vs non-existing collections).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from qdrant_client import models
from qdrant_client.models import ScoredPoint

import vector_classifier.main as app_main
from vector_classifier.main import app, get_engine, set_engine
from vector_classifier.vectors.classifier import (
    VectorClassifierEngine,
    extract_header_content,
)
from vector_classifier.vectors.client import OllamaEmbeddingClient
from vector_classifier.vectors.qdrant import (
    PAYLOAD_INDEXES,
    QdrantStorageError,
    QdrantStorageManager,
    _build_filter,
)


# ===========================================================================
# 1. Empty OCR texts, missing titles, whitespace-only texts
# ===========================================================================
@pytest.mark.asyncio
async def test_classify_empty_and_whitespace_content():
    """Verify classification engine handles empty strings and whitespace-only text gracefully."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    # Returns dummy zero vectors for empty/fallback input
    dummy_vec = [0.0] * 8
    mock_ollama.get_document_embeddings.return_value = (dummy_vec, dummy_vec)

    test_anchors = {
        "Rechnung": [1.0] + [0.0] * 7,
        "steuer:werbungskosten": [1.0] + [0.0] * 7,
        "Energie": [1.0] + [0.0] * 7,
    }

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=None,
        anchor_embeddings=test_anchors,
    )

    # Empty string
    res_empty = await engine.classify(content="", filename="", sha256="")
    assert res_empty.predicted_doc_type is None
    assert res_empty.is_tax_relevant is False
    assert res_empty.predicted_tags == []
    assert res_empty.tax_confidence == 0.0

    # Whitespace-only string
    res_whitespace = await engine.classify(
        content="   \n\t\r   \n   ",
        filename="",
        sha256="",
    )
    assert res_whitespace.predicted_doc_type is None
    assert res_whitespace.is_tax_relevant is False
    assert res_whitespace.predicted_tags == []


@pytest.mark.asyncio
async def test_classify_missing_title_and_explicit_header():
    """Verify classify behaves correctly when title/filename is missing and header_content is provided."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_ollama.get_document_embeddings.return_value = ([0.1] * 8, [0.2] * 8)

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=None,
        anchor_embeddings={},
    )

    result = await engine.classify(
        content="Full body text of the document",
        header_content="Explicit Header Text",
    )
    assert result is not None
    # Verify get_document_embeddings was called with the explicit header
    mock_ollama.get_document_embeddings.assert_awaited_with(
        "Explicit Header Text",
        "Full body text of the document",
    )


@pytest.mark.asyncio
async def test_api_classify_empty_payload():
    """Verify REST API POST /api/v1/classify succeeds on empty OCR content."""
    mock_engine = AsyncMock(spec=VectorClassifierEngine)
    from vector_classifier.models.schemas import VectorClassificationResult

    mock_engine.classify.return_value = VectorClassificationResult(
        predicted_doc_type=None,
        doc_type_confidence=0.0,
        predicted_tags=[],
        tag_scores={},
        is_tax_relevant=False,
        tax_confidence=0.0,
        tax_category=None,
        tax_review_required=False,
        matched_anchors=[],
    )
    set_engine(mock_engine)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/classify",
            json={"content": "  ", "filename": "", "sha256": ""},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_doc_type"] is None
        assert data["is_tax_relevant"] is False


# ===========================================================================
# 2. Very long texts (>8,192 tokens / multi-page documents) & Splitting
# ===========================================================================
def test_extract_header_content_long_multipage_text():
    """Verify extract_header_content truncates long texts at suitable paragraph boundaries."""
    # Build a simulated 10-page document (~40,000 characters)
    pages = []
    for i in range(1, 11):
        pages.append(f"--- Page {i} ---\n" + ("Lorem ipsum dolor sit amet. " * 150))
    long_doc = "\n\n".join(pages)
    assert len(long_doc) > 30000

    header = extract_header_content(long_doc, max_chars=3000)
    assert len(header) <= 3000
    assert header.startswith("--- Page 1 ---")
    # Must cut on a newline cleanly
    assert not header.endswith("\n")
    # Verify the remaining pages are excluded from header
    assert "--- Page 5 ---" not in header


def test_extract_header_content_various_lengths():
    """Test header extraction boundary conditions with various string shapes."""
    # Text shorter than max_chars
    short = "Briefkopf\nEmpfänger\nBetreff"
    assert extract_header_content(short, max_chars=3000) == short

    # Text exactly at boundary
    exact = "A" * 3000
    assert extract_header_content(exact, max_chars=3000) == exact

    # Text exceeding boundary without newline after 500 chars
    long_no_newline = "B" * 5000
    header_unbroken = extract_header_content(long_no_newline, max_chars=3000)
    assert len(header_unbroken) == 3000
    assert header_unbroken == "B" * 3000

    # Text with newline before 500 chars but none between 500 and 3000
    text_early_newline = "Start\n" + ("C" * 4000)
    header_early = extract_header_content(text_early_newline, max_chars=3000)
    assert len(header_early) == 3000

    # Text with newline between 500 and 3000
    text_good_break = ("D" * 2000) + "\n" + ("E" * 2000)
    header_good = extract_header_content(text_good_break, max_chars=3000)
    assert header_good == "D" * 2000


@pytest.mark.asyncio
async def test_classify_splits_long_document_correctly():
    """Verify classify separates header and content for very long documents."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_ollama.get_document_embeddings.return_value = ([0.0] * 8, [0.0] * 8)

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=None,
        anchor_embeddings={},
    )

    long_content = "Header Line 1\nHeader Line 2\n" + (
        "Body Paragraph Content\n" * 2000
    )
    await engine.classify(content=long_content)

    mock_ollama.get_document_embeddings.assert_awaited_once()
    called_header, called_content = mock_ollama.get_document_embeddings.call_args.args
    assert len(called_header) <= 3000
    assert called_header.startswith("Header Line 1")
    assert called_content == long_content


# ===========================================================================
# 3. Precise boundary value tests for three-zone tax logic
# ===========================================================================
def _make_unit_vector(similarity: float, dim: int = 8) -> list[float]:
    import math

    sim = max(-1.0, min(1.0, similarity))
    vec = [0.0] * dim
    vec[0] = sim
    vec[1] = math.sqrt(max(0.0, 1.0 - sim * sim))
    return vec


@pytest.mark.parametrize(
    ("score", "expected_relevant", "expected_review", "expect_category"),
    [
        (0.5100, False, False, False),
        (0.5199, False, False, False),
        (0.5200, True, True, True),
        (0.5201, True, True, True),
        (0.5900, True, True, True),
        (0.5999, True, True, True),
        (0.6000, True, False, True),
        (0.6001, True, False, True),
    ],
)
@pytest.mark.asyncio
async def test_tax_three_zone_precise_boundaries(
    score: float,
    expected_relevant: bool,
    expected_review: bool,
    expect_category: bool,
):
    """Test tax classification at exact sub-cent boundary thresholds (0.5199, 0.5200, 0.5999, 0.6000)."""
    ref_vec = [1.0] + [0.0] * 7
    anchor_dict = {
        "steuer:werbungskosten": ref_vec,
        "steuer:haushaltsnah-35a": [0.0, 0.0, 1.0] + [0.0] * 5,
        "steuer:sonderausgaben": [0.0, 0.0, 0.0, 1.0] + [0.0] * 4,
        "steuer:krankheit-aussergewoehnlich": [0.0] * 4 + [1.0] + [0.0] * 3,
        "steuer:vermietung-anlage-v": [0.0] * 5 + [1.0] + [0.0] * 2,
    }

    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    content_vec = _make_unit_vector(score, dim=8)
    mock_ollama.get_document_embeddings.return_value = ([0.0] * 8, content_vec)

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=None,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Steuer Dokument")

    assert result.is_tax_relevant == expected_relevant
    assert result.tax_review_required == expected_review
    if expect_category:
        assert result.tax_category == "steuer:werbungskosten"
    else:
        assert result.tax_category is None


@pytest.mark.parametrize(
    ("knn_score", "expected_relevant", "expected_review", "expect_category"),
    [
        (0.5199, False, False, False),
        (0.5200, True, True, True),
        (0.5999, True, True, True),
        (0.6000, True, False, True),
    ],
)
@pytest.mark.asyncio
async def test_knn_tax_boost_boundaries(
    knn_score: float,
    expected_relevant: bool,
    expected_review: bool,
    expect_category: bool,
):
    """Test that k-NN neighbor scores trigger three-zone transitions correctly at boundaries."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_ollama.get_document_embeddings.return_value = ([0.0] * 8, [0.0] * 8)

    # Low anchor baseline
    anchor_dict = {
        "steuer:werbungskosten": _make_unit_vector(0.50, dim=8),
        "steuer:haushaltsnah-35a": _make_unit_vector(0.50, dim=8),
        "steuer:sonderausgaben": _make_unit_vector(0.50, dim=8),
        "steuer:krankheit-aussergewoehnlich": _make_unit_vector(0.50, dim=8),
        "steuer:vermietung-anlage-v": _make_unit_vector(0.50, dim=8),
    }

    mock_qdrant = AsyncMock(spec=QdrantStorageManager)
    scored_point = ScoredPoint(
        id="pt-1",
        version=1,
        score=knn_score,
        payload={"is_tax_relevant": True, "tax_category": "steuer:haushaltsnah-35a"},
        vector=None,
    )
    mock_qdrant.search_by_vector.return_value = [scored_point]

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=mock_qdrant,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Handwerkerrechnung")

    assert result.is_tax_relevant == expected_relevant
    assert result.tax_review_required == expected_review
    if expect_category:
        assert result.tax_category == "steuer:haushaltsnah-35a"
    else:
        assert result.tax_category is None


# ===========================================================================
# 4. Qdrant Storage Manager edge cases
# ===========================================================================
@pytest.mark.asyncio
async def test_qdrant_unreachable_host_handling():
    """Verify Qdrant operations wrap network/connection failures in QdrantStorageError."""
    mock_client = AsyncMock()
    mock_client.collection_exists.side_effect = ConnectionRefusedError(
        "Host unreachable"
    )
    mock_client.upsert.side_effect = ConnectionResetError("Connection reset")
    mock_client.query_points.side_effect = TimeoutError("Request timed out")
    mock_client.get_collections.side_effect = ConnectionRefusedError("Unreachable")

    manager = QdrantStorageManager(client=mock_client)

    # Health check returns False without raising
    assert await manager.health_check() is False

    # ensure_collection raises QdrantStorageError
    with pytest.raises(QdrantStorageError, match="Failed to ensure collection"):
        await manager.ensure_collection()

    # upsert_document raises QdrantStorageError
    with pytest.raises(QdrantStorageError, match="Failed to upsert document"):
        await manager.upsert_document(
            point_id="pt-1",
            header_vector=[0.0] * 1024,
            content_vector=[0.0] * 1024,
            payload={},
        )

    # search_by_vector raises QdrantStorageError
    with pytest.raises(QdrantStorageError, match="Vector search failed"):
        await manager.search_by_vector(vector_name="content", vector=[0.0] * 1024)


def test_build_filter_edge_cases():
    """Verify _build_filter handles edge cases like empty dict, None, and varied payload types."""
    # None returns None
    assert _build_filter(None) is None

    # Empty dict returns Filter with empty must list
    f_empty = _build_filter({})
    assert isinstance(f_empty, models.Filter)
    assert f_empty.must == []

    # Already a Filter
    existing = models.Filter(must=[])
    assert _build_filter(existing) is existing

    # Criteria with list and scalar
    criteria = {
        "tags": ["A", "B"],
        "count": 5,
        "is_active": False,
    }
    f = _build_filter(criteria)
    assert f is not None
    assert isinstance(f.must, list)
    assert len(f.must) == 3


@pytest.mark.asyncio
async def test_qdrant_collection_creation_vs_existing():
    """Verify ensure_collection creates payload indexes only when collection did not exist."""
    mock_client = AsyncMock()

    # Case 1: Collection does not exist
    mock_client.collection_exists.return_value = False
    mock_client.create_collection.return_value = True
    mock_client.create_payload_index.return_value = MagicMock()

    manager = QdrantStorageManager(client=mock_client, collection="new_docs")
    await manager.ensure_collection()

    assert mock_client.create_collection.await_count == 1
    assert mock_client.create_payload_index.await_count == len(PAYLOAD_INDEXES)

    # Case 2: Collection exists already
    mock_client.reset_mock()
    mock_client.collection_exists.return_value = True

    await manager.ensure_collection()
    mock_client.create_collection.assert_not_awaited()
    mock_client.create_payload_index.assert_not_awaited()


@pytest.mark.asyncio
async def test_qdrant_client_property_and_setter():
    """Verify lazy initialization and property setter of AsyncQdrantClient."""
    from unittest.mock import patch

    with patch("vector_classifier.vectors.qdrant.AsyncQdrantClient") as mock_cls:
        manager = QdrantStorageManager(host="localhost", port=6333)
        assert manager._client is None
        # Access property -> lazily initializes client via AsyncQdrantClient constructor
        client = manager.client
        assert client is mock_cls.return_value
        assert manager._client is client
        mock_cls.assert_called_once_with(
            host="localhost", port=6333, api_key=None, timeout=60
        )

    # Setter overrides client
    new_mock = AsyncMock()
    manager.client = new_mock
    assert manager.client is new_mock
    assert manager._external_client is True


@pytest.mark.asyncio
async def test_qdrant_search_neither_query_points_nor_search():
    """Verify search_by_vector raises QdrantStorageError when client has no search method."""
    mock_client = MagicMock(spec=[])  # Client with neither query_points nor search
    manager = QdrantStorageManager(client=mock_client)

    with pytest.raises(
        QdrantStorageError,
        match="Underlying client does not support query_points or search",
    ):
        await manager.search_by_vector(vector_name="content", vector=[0.0] * 1024)


# ===========================================================================
# 5. Coverage completion helper tests
# ===========================================================================
def test_get_engine_lazy_instantiation():
    """Verify get_engine instantiates VectorClassifierEngine if _engine is None."""
    set_engine(None)
    assert app_main._engine is None
    engine = get_engine()
    assert isinstance(engine, VectorClassifierEngine)
    assert app_main._engine is engine
    set_engine(None)


@pytest.mark.asyncio
async def test_classify_doc_type_no_anchors():
    """Verify _classify_doc_type returns (None, 0.0, []) when anchor catalog is empty."""
    from unittest.mock import patch

    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=None,
        anchor_embeddings={},
    )
    with patch(
        "vector_classifier.vectors.classifier.get_document_type_anchors",
        return_value=[],
    ):
        pred_type, conf, matched = await engine._classify_doc_type([0.1] * 8)
        assert pred_type is None
        assert conf == 0.0
        assert matched == []


@pytest.mark.asyncio
async def test_knn_tag_votes_empty_neighbors():
    """Verify _apply_knn_tag_votes returns cleanly when Qdrant returns no neighbors."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_qdrant = AsyncMock(spec=QdrantStorageManager)
    mock_qdrant.search_by_vector.return_value = []

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=mock_qdrant,
        anchor_embeddings={},
    )
    qualifying: set[str] = set()
    scores: dict[str, float] = {}
    await engine._apply_knn_tag_votes([0.1] * 8, qualifying, scores)
    assert qualifying == set()
    assert scores == {}
