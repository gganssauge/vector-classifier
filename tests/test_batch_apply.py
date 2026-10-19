"""Unit tests for BatchApplyRunner and batch application to Paperless."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from rich.console import Console

from vector_classifier.batch_apply import BatchApplyRunner
from vector_classifier.vectors.classifier import VectorClassifierEngine
from vector_classifier.vectors.qdrant import QdrantStorageManager


@pytest.fixture
def mock_qdrant():
    """Mock QdrantStorageManager."""
    mgr = AsyncMock(spec=QdrantStorageManager)
    mgr.collection = "paperless_documents"
    mgr.client = AsyncMock()
    return mgr


@pytest.fixture
def mock_engine():
    """Mock VectorClassifierEngine."""
    engine = AsyncMock(spec=VectorClassifierEngine)
    engine.initialize_anchor_embeddings = AsyncMock()
    return engine


def test_batch_apply_init_and_headers(mock_qdrant, mock_engine):
    """Test initialization and headers formatting."""
    runner = BatchApplyRunner(
        paperless_url="http://paperless.test/",
        paperless_token="secret-token-123",
        qdrant_manager=mock_qdrant,
        engine=mock_engine,
    )
    assert runner.paperless_url == "http://paperless.test"
    assert runner.paperless_token == "secret-token-123"
    assert runner._get_headers() == {"Authorization": "Token secret-token-123"}


def test_batch_apply_default_init():
    """Test default initialization values."""
    runner = BatchApplyRunner()
    assert runner.paperless_token is not None
    assert runner.qdrant is not None
    assert runner.engine is not None


@pytest.mark.asyncio
async def test_get_or_create_tag_existing():
    """Test retrieving existing tag ID."""
    runner = BatchApplyRunner(paperless_url="http://test", paperless_token="tok")
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [{"id": 42, "name": "Doctype-Review"}, {"id": 43, "name": "Energie"}]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_client.get.return_value = mock_resp

    tag_id = await runner.get_or_create_tag(mock_client, "Doctype-Review")
    assert tag_id == 42
    mock_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_get_or_create_tag_create():
    """Test creating a new tag when not found."""
    runner = BatchApplyRunner(paperless_url="http://test", paperless_token="tok")
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    get_resp = MagicMock()
    get_resp.json.return_value = {"results": []}
    get_resp.raise_for_status = MagicMock()
    mock_client.get.return_value = get_resp

    post_resp = MagicMock()
    post_resp.json.return_value = {"id": 99, "name": "NewTag"}
    post_resp.raise_for_status = MagicMock()
    mock_client.post.return_value = post_resp

    tag_id = await runner.get_or_create_tag(mock_client, "NewTag", color="#ff0000")
    assert tag_id == 99
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_load_metadata():
    """Test loading document types and tags metadata."""
    runner = BatchApplyRunner(paperless_url="http://test", paperless_token="tok")
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    dt_resp = MagicMock()
    dt_resp.json.return_value = {
        "results": [{"id": 1, "name": "Rechnung"}, {"id": 2, "name": "Vertrag"}]
    }
    dt_resp.raise_for_status = MagicMock()

    tag_resp = MagicMock()
    tag_resp.json.return_value = {
        "results": [
            {"id": 10, "name": "Doctype-Review"},
            {"id": 11, "name": "Steuer-Review"},
            {"id": 12, "name": "steuerrelevant"},
            {"id": 13, "name": "Energie"},
        ]
    }
    tag_resp.raise_for_status = MagicMock()

    mock_client.get.side_effect = [dt_resp, tag_resp]

    (
        doc_type_map,
        tag_map,
        dt_rev_id,
        st_rev_id,
        st_rel_id,
        macro_ids,
    ) = await runner._load_metadata(mock_client, dry_run=True)

    assert doc_type_map == {"Rechnung": 1, "Vertrag": 2}
    assert dt_rev_id == 10
    assert st_rev_id == 11
    assert st_rel_id == 12
    assert "Energie" in macro_ids
    assert macro_ids["Energie"] == 13


@pytest.mark.asyncio
async def test_evaluate_doctype_curated():
    """Test evaluate doctype with curated flag."""
    runner = BatchApplyRunner()
    green: dict[str, list[int]] = {}
    yellow: list[int] = []
    groundtruth: list[int] = []

    # Curated matches original -> no green change, but added to groundtruth
    point = MagicMock()
    payload = {
        "is_doctype_curated": True,
        "doctype_curated": "Rechnung",
        "doc_type_original": "Rechnung",
    }
    await runner._evaluate_doctype(
        point, payload, 101, [0.1], [0.2], green, yellow, groundtruth
    )
    assert 101 in groundtruth
    assert green == {}
    assert yellow == []

    # Curated differs from original -> added to groundtruth AND green assignment
    payload2 = {
        "is_doctype_curated": True,
        "doctype_curated": "Mahnung",
        "doc_type_original": "Rechnung",
    }
    await runner._evaluate_doctype(
        point, payload2, 102, [0.1], [0.2], green, yellow, groundtruth
    )
    assert 102 in groundtruth
    assert green["Mahnung"] == [102]


@pytest.mark.asyncio
async def test_evaluate_doctype_uncurated_green_and_yellow(mock_engine):
    """Test evaluate doctype with uncurated points (green vs yellow)."""
    runner = BatchApplyRunner(engine=mock_engine)
    green: dict[str, list[int]] = {}
    yellow: list[int] = []
    groundtruth: list[int] = []

    # Uncurated, high confidence, not review required
    mock_engine._classify_doc_type.return_value = (
        "Versicherungspolice",
        0.88,
        [{"name": "Versicherungspolice", "review_required": False, "knn_match": None}],
    )
    point = MagicMock()
    point.id = "uuid-1"
    payload = {"doc_type_original": "Sonstiges"}
    await runner._evaluate_doctype(
        point, payload, 201, [0.1], [0.2], green, yellow, groundtruth
    )
    assert green["Versicherungspolice"] == [201]
    assert yellow == []

    # Uncurated, review required
    mock_engine._classify_doc_type.return_value = (
        "Rechnung",
        0.58,
        [{"name": "Rechnung", "review_required": True, "knn_match": None}],
    )
    await runner._evaluate_doctype(
        point, payload, 202, [0.1], [0.2], green, yellow, groundtruth
    )
    assert 202 in yellow


@pytest.mark.asyncio
async def test_evaluate_tags_and_tax(mock_engine):
    """Test evaluate tags and tax relevance."""
    runner = BatchApplyRunner(engine=mock_engine)
    tags_to_add: dict[str, list[int]] = {}
    tax_conf: list[int] = []
    tax_rev: list[int] = []

    mock_engine._classify_thematic_tags.return_value = (
        ["Energie", "Bank"],
        [("Energie", 0.85), ("Bank", 0.82)],
        [],
    )
    mock_engine._classify_tax_relevance.return_value = (
        True,
        0.88,
        "steuer:werbungskosten",
        False,
        [],
    )

    # Point without c_vec -> returns early
    await runner._evaluate_tags_and_tax({}, 300, None, tags_to_add, tax_conf, tax_rev)
    assert tags_to_add == {}

    # Point with string tag
    payload = {"tags_original": "Sonstiges", "is_tax_curated": False}
    await runner._evaluate_tags_and_tax(
        payload, 301, [0.1] * 8, tags_to_add, tax_conf, tax_rev
    )
    assert tags_to_add["Energie"] == [301]
    assert tags_to_add["Bank"] == [301]
    assert tax_conf == [301]
    assert tax_rev == []

    # Point with tax review
    mock_engine._classify_tax_relevance.return_value = (
        False,
        0.55,
        None,
        True,
        [],
    )
    payload2 = {"tags_original": ["Energie"], "is_tax_curated": False}
    await runner._evaluate_tags_and_tax(
        payload2, 302, [0.1] * 8, tags_to_add, tax_conf, tax_rev
    )
    assert 302 not in tags_to_add["Energie"]  # Already in existing_tags
    assert tags_to_add["Bank"] == [301, 302]
    assert tax_rev == [302]


@pytest.mark.asyncio
async def test_evaluate_point(mock_engine):
    """Test evaluate point wrapper."""
    runner = BatchApplyRunner(engine=mock_engine)
    green: dict[str, list[int]] = {}
    yellow: list[int] = []
    groundtruth: list[int] = []
    tags_to_add: dict[str, list[int]] = {}
    tax_conf: list[int] = []
    tax_rev: list[int] = []

    # Invalid record (missing paperless_id)
    r_invalid = MagicMock()
    r_invalid.payload = {}
    ok = await runner._evaluate_point(
        r_invalid, green, yellow, groundtruth, tags_to_add, tax_conf, tax_rev
    )
    assert ok is False

    # Valid record with dict vector
    r_valid = MagicMock()
    r_valid.id = "uuid-123"
    r_valid.payload = {"paperless_id": 401, "doc_type_original": "Rechnung"}
    r_valid.vector = {"header": [0.1] * 8, "content": [0.2] * 8}

    mock_engine._classify_doc_type.return_value = ("Rechnung", 0.9, [])
    mock_engine._classify_thematic_tags.return_value = ([], [], [])
    mock_engine._classify_tax_relevance.return_value = (False, 0.0, None, False, [])

    ok = await runner._evaluate_point(
        r_valid, green, yellow, groundtruth, tags_to_add, tax_conf, tax_rev
    )
    assert ok is True


@pytest.mark.asyncio
async def test_post_bulk_chunks():
    """Test resilient chunked posting to Paperless bulk API."""
    runner = BatchApplyRunner(paperless_url="http://test", paperless_token="tok")
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_resp
    con = Console(quiet=True)

    # Empty list
    await runner._post_bulk_chunks(
        mock_client, con, [], "modify_tags", {}, "Test empty"
    )
    mock_client.post.assert_not_called()

    # 75 IDs -> 2 chunks (50 + 25)
    ids = list(range(1, 76))
    await runner._post_bulk_chunks(
        mock_client, con, ids, "modify_tags", {"add_tags": [1]}, "Tagging"
    )
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_scroll_and_evaluate_all(mock_qdrant, mock_engine):
    """Test scrolling through Qdrant records."""
    runner = BatchApplyRunner(qdrant_manager=mock_qdrant, engine=mock_engine)

    rec1 = MagicMock()
    rec1.id = "1"
    rec1.payload = {"paperless_id": 501}
    rec1.vector = {"header": [0.1] * 8, "content": [0.2] * 8}

    rec2 = MagicMock()
    rec2.id = "2"
    rec2.payload = {"paperless_id": 502}
    rec2.vector = {"header": [0.1] * 8, "content": [0.2] * 8}

    mock_qdrant.client.scroll.side_effect = [
        ([rec1, rec2], "page-2"),
        ([], None),
    ]

    mock_engine._classify_doc_type.return_value = ("Vertrag", 0.9, [])
    mock_engine._classify_thematic_tags.return_value = ([], [], [])
    mock_engine._classify_tax_relevance.return_value = (False, 0.0, None, False, [])

    green: dict[str, list[int]] = {}
    yellow: list[int] = []
    groundtruth: list[int] = []
    tags_to_add: dict[str, list[int]] = {}
    tax_conf: list[int] = []
    tax_rev: list[int] = []

    total = await runner._scroll_and_evaluate_all(
        limit=10,
        doctype_green_assign=green,
        doctype_yellow_review=yellow,
        doctype_groundtruth=groundtruth,
        tags_to_add=tags_to_add,
        tax_confident=tax_conf,
        tax_review=tax_rev,
    )
    assert total == 2


@pytest.mark.asyncio
async def test_batch_apply_run_dry_run_and_live(mock_qdrant, mock_engine):
    """Test complete run in dry-run and live modes."""
    runner = BatchApplyRunner(qdrant_manager=mock_qdrant, engine=mock_engine)

    rec = MagicMock()
    rec.id = "rec-1"
    rec.payload = {"paperless_id": 601, "doc_type_original": "Sonstiges"}
    rec.vector = {"header": [0.1] * 8, "content": [0.2] * 8}

    mock_qdrant.client.scroll.return_value = ([rec], None)
    mock_engine._classify_doc_type.return_value = (
        "Vertrag",
        0.88,
        [{"name": "Vertrag", "review_required": False, "knn_match": None}],
    )
    mock_engine._classify_thematic_tags.return_value = (["Energie"], [], [])
    mock_engine._classify_tax_relevance.return_value = (
        True,
        0.9,
        "steuer:35a",
        False,
        [],
    )

    # Mock metadata responses
    con = Console(quiet=True)

    # Use respx or mock httpx in run
    from unittest.mock import patch

    with patch("httpx.AsyncClient") as mock_client_cls:
        client_inst = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = client_inst

        dt_resp = MagicMock()
        dt_resp.json.return_value = {"results": [{"id": 5, "name": "Vertrag"}]}
        dt_resp.raise_for_status = MagicMock()

        tag_resp = MagicMock()
        tag_resp.json.return_value = {
            "results": [
                {"id": 20, "name": "Doctype-Review"},
                {"id": 21, "name": "Steuer-Review"},
                {"id": 22, "name": "steuerrelevant"},
                {"id": 23, "name": "Energie"},
            ]
        }
        tag_resp.raise_for_status = MagicMock()

        bulk_resp = MagicMock()
        bulk_resp.raise_for_status = MagicMock()

        def mock_get(url, *args, **kwargs):
            if "document_types" in str(url):
                return dt_resp
            return tag_resp

        client_inst.get.side_effect = mock_get
        client_inst.post.return_value = bulk_resp

        # 1. Dry run
        res_dry = await runner.run(dry_run=True, console=con)
        assert res_dry["total_processed"] == 1
        assert res_dry["doctype_green"] == 1
        assert res_dry["tags_added"] == 1
        assert res_dry["tax_confident"] == 1
        # In dry run, bulk post is never called
        client_inst.post.assert_not_called()

        # 2. Live run
        res_live = await runner.run(dry_run=False, console=con)
        assert res_live["total_processed"] == 1
        # Bulk post is called for doctype, yellow review, tags, tax
        assert client_inst.post.call_count >= 1
