"""Unit tests for vector classification engine, cosine similarity, and k-NN integration."""

import math
from unittest.mock import AsyncMock

import pytest
from qdrant_client.models import ScoredPoint

from vector_classifier.vectors.anchors import (
    get_document_type_anchors,
    get_tax_anchors,
    get_thematic_tag_anchors,
)
from vector_classifier.vectors.classifier import (
    VectorClassifierEngine,
    cosine_similarity,
    extract_header_content,
)
from vector_classifier.vectors.client import OllamaEmbeddingClient
from vector_classifier.vectors.qdrant import QdrantStorageError, QdrantStorageManager


# ---------------------------------------------------------------------------
# Cosine Similarity Tests
# ---------------------------------------------------------------------------
def test_cosine_similarity_identical_vectors():
    """Identical vectors should yield cosine similarity of 1.0."""
    vec = [1.0, 2.0, 3.0]
    sim = cosine_similarity(vec, vec)
    assert math.isclose(sim, 1.0, rel_tol=1e-5)


def test_cosine_similarity_orthogonal_vectors():
    """Orthogonal vectors should yield cosine similarity of 0.0."""
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    assert math.isclose(cosine_similarity(v1, v2), 0.0, abs_tol=1e-5)


def test_cosine_similarity_opposing_vectors():
    """Opposing vectors should yield cosine similarity of -1.0."""
    v1 = [1.0, 0.0]
    v2 = [-1.0, 0.0]
    assert math.isclose(cosine_similarity(v1, v2), -1.0, rel_tol=1e-5)


def test_cosine_similarity_edge_cases():
    """Zero vectors, empty vectors, or length mismatches return 0.0."""
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# Header Content Extraction Tests
# ---------------------------------------------------------------------------
def test_extract_header_content():
    """Verify header extraction clips long texts and preserves short texts."""
    assert extract_header_content("") == ""
    short_text = "Rechnung Nr. 12345\nDatum: 2026-03-01\nBetrag: 100 EUR"
    assert extract_header_content(short_text) == short_text

    long_text = ("Abschnitt A\n" * 300) + ("Abschnitt B\n" * 300)
    extracted = extract_header_content(long_text, max_chars=1000)
    assert len(extracted) <= 1000
    assert extracted.endswith("Abschnitt A")

    unbroken = "x" * 2000
    extracted_unbroken = extract_header_content(unbroken, max_chars=1000)
    assert len(extracted_unbroken) == 1000


# ---------------------------------------------------------------------------
# Helper: Synthetic Vector Builder
# ---------------------------------------------------------------------------
def make_unit_vector_with_similarity(similarity: float, dim: int = 16) -> list[float]:
    """Generate a unit vector having exact cosine similarity with [1, 0, 0, ...]."""
    sim_clipped = max(-1.0, min(1.0, similarity))
    vec = [0.0] * dim
    vec[0] = sim_clipped
    vec[1] = math.sqrt(max(0.0, 1.0 - sim_clipped * sim_clipped))
    return vec


def make_ref_vector(dim: int = 16) -> list[float]:
    """Reference basis vector [1, 0, 0, ...]."""
    vec = [0.0] * dim
    vec[0] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Document Type Classification Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doctype_classification_confident():
    """High similarity (>= 0.80) assigns predicted_doc_type confidently."""
    ref_vec = make_ref_vector(8)
    anchor_dict: dict[str, list[float]] = {}
    for anchor in get_document_type_anchors():
        if anchor.name == "Rechnung":
            anchor_dict[anchor.name] = ref_vec
        else:
            other = [0.0] * 8
            other[2] = 1.0
            anchor_dict[anchor.name] = other

    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    # Header vector aligns with Rechnung (similarity 0.95)
    query_header = make_unit_vector_with_similarity(0.95, dim=8)
    content_vec = [0.0] * 8
    mock_ollama.get_document_embeddings.return_value = (query_header, content_vec)

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=None,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Dummy Inhalt", filename="test.pdf")

    assert result.predicted_doc_type == "Rechnung"
    assert result.doc_type_confidence == 0.95
    assert any(a["name"] == "Rechnung" for a in result.matched_anchors)


@pytest.mark.asyncio
async def test_doctype_classification_low_confidence():
    """Low similarity (< 0.50) results in predicted_doc_type=None."""
    anchor_dict: dict[str, list[float]] = {}
    for anchor in get_document_type_anchors():
        anchor_dict[anchor.name] = make_ref_vector(8)

    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    # Header vector has only 0.35 similarity (< 0.50)
    query_header = make_unit_vector_with_similarity(0.35, dim=8)
    content_vec = [0.0] * 8
    mock_ollama.get_document_embeddings.return_value = (query_header, content_vec)

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=None,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Dummy Inhalt")

    assert result.predicted_doc_type is None
    assert result.doc_type_confidence == 0.35


@pytest.mark.asyncio
async def test_doctype_classification_ambiguous_margin():
    """Close scores between top-1 and top-2 (< 0.05) are flagged as ambiguous."""
    anchor_dict: dict[str, list[float]] = {}
    for anchor in get_document_type_anchors():
        if anchor.name == "Rechnung":
            # Top-1: similarity 0.86
            anchor_dict[anchor.name] = make_unit_vector_with_similarity(0.86, dim=8)
        elif anchor.name == "Abrechnung":
            # Top-2: similarity 0.83 (margin = 0.03 < 0.05)
            anchor_dict[anchor.name] = make_unit_vector_with_similarity(0.83, dim=8)
        else:
            anchor_dict[anchor.name] = [0.0] * 8

    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    query_header = make_ref_vector(8)
    content_vec = [0.0] * 8
    mock_ollama.get_document_embeddings.return_value = (query_header, content_vec)

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=None,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Rechnung Abrechnung")

    assert result.predicted_doc_type == "Rechnung"
    assert result.doc_type_confidence == 0.86
    doc_anchor_entry = next(
        a for a in result.matched_anchors if a.get("category") == "doc_type"
    )
    assert doc_anchor_entry["ambiguous"] is True


# ---------------------------------------------------------------------------
# Tax Three-Zone Boundary Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("score", "expected_relevant", "expected_review", "expect_category"),
    [
        (0.51, False, False, False),
        (0.52, True, True, True),
        (0.59, True, True, True),
        (0.60, True, False, True),
    ],
)
@pytest.mark.asyncio
async def test_tax_three_zone_boundaries(
    score: float,
    expected_relevant: bool,
    expected_review: bool,
    expect_category: bool,
):
    """Test three-zone tax relevance boundary conditions (0.51, 0.52, 0.59, 0.60)."""
    ref_vec = make_ref_vector(8)
    anchor_dict: dict[str, list[float]] = {}
    for a in get_tax_anchors():
        if a.name == "steuer:werbungskosten":
            anchor_dict[a.name] = ref_vec
        else:
            other = [0.0] * 8
            other[3] = 1.0
            anchor_dict[a.name] = other

    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    header_vec = [0.0] * 8
    content_vec = make_unit_vector_with_similarity(score, dim=8)
    mock_ollama.get_document_embeddings.return_value = (header_vec, content_vec)

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=None,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Steuerlicher Text")

    assert result.is_tax_relevant == expected_relevant
    assert result.tax_review_required == expected_review
    assert math.isclose(result.tax_confidence, score, abs_tol=1e-4)
    if expect_category:
        assert result.tax_category == "steuer:werbungskosten"
    else:
        assert result.tax_category is None


# ---------------------------------------------------------------------------
# Thematic Tag Multi-Label Thresholding Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_thematic_tag_multilabel_thresholding():
    """Tags with similarity >= 0.50 are assigned; tags below are excluded."""
    ref_vec = make_ref_vector(8)
    anchor_dict: dict[str, list[float]] = {}
    for a in get_thematic_tag_anchors():
        if a.name == "Energie":
            # Similarity = 0.85 (qualifies >= 0.50)
            anchor_dict[a.name] = make_unit_vector_with_similarity(0.85, dim=8)
        elif a.name == "Fahrzeug":
            # Similarity = 0.55 (qualifies >= 0.50)
            anchor_dict[a.name] = make_unit_vector_with_similarity(0.55, dim=8)
        elif a.name == "Telekommunikation":
            # Similarity = 0.45 (excluded < 0.50)
            anchor_dict[a.name] = make_unit_vector_with_similarity(0.45, dim=8)
        else:
            anchor_dict[a.name] = [0.0] * 8

    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    header_vec = [0.0] * 8
    # Content matches ref_vec so dot product with make_unit_vector_with_similarity(s) is s
    content_vec = ref_vec
    mock_ollama.get_document_embeddings.return_value = (header_vec, content_vec)

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=None,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Strom und Auto")

    assert "Energie" in result.predicted_tags
    assert "Fahrzeug" in result.predicted_tags
    assert "Telekommunikation" not in result.predicted_tags
    assert result.predicted_tags == ["Energie", "Fahrzeug"]
    assert result.tag_scores["Energie"] == 0.85
    assert result.tag_scores["Fahrzeug"] == 0.55
    assert result.tag_scores["Telekommunikation"] == 0.45


# ---------------------------------------------------------------------------
# k-NN Integration Tests with Mock Qdrant
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_knn_tax_boost_and_category():
    """k-NN neighbor with higher tax score overrides lower anchor score."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_ollama.get_document_embeddings.return_value = ([0.0] * 8, [0.0] * 8)

    # Low anchor score (0.50)
    anchor_dict: dict[str, list[float]] = {}
    for a in get_tax_anchors():
        anchor_dict[a.name] = make_unit_vector_with_similarity(0.50, dim=8)

    mock_qdrant = AsyncMock(spec=QdrantStorageManager)
    scored_point = ScoredPoint(
        id="dummy-point-id",
        version=1,
        score=0.89,
        payload={
            "is_tax_relevant": True,
            "tax_category": "steuer:haushaltsnah-35a",
        },
        vector=None,
    )
    mock_qdrant.search_by_vector.return_value = [scored_point]

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=mock_qdrant,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Handwerkerrechnung")

    assert result.is_tax_relevant is True
    assert result.tax_review_required is False
    assert result.tax_confidence == 0.89
    assert result.tax_category == "steuer:haushaltsnah-35a"


@pytest.mark.asyncio
async def test_knn_thematic_tag_support():
    """Tags supported by >= 2 neighbors in Qdrant are assigned even if anchor < 0.78."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_ollama.get_document_embeddings.return_value = ([0.0] * 8, [0.0] * 8)

    # All anchors have low similarity (0.50)
    anchor_dict: dict[str, list[float]] = {}
    for a in get_thematic_tag_anchors():
        anchor_dict[a.name] = make_unit_vector_with_similarity(0.50, dim=8)

    mock_qdrant = AsyncMock(spec=QdrantStorageManager)

    # Search for tax returns empty; search for tags returns 3 neighbors with tag "Gesundheit"
    async def fake_search(vector_name: str, **kwargs):
        if kwargs.get("filter_criteria"):
            return []
        p1 = ScoredPoint(
            id="1",
            version=1,
            score=0.8,
            payload={"tags": ["Gesundheit"]},
            vector=None,
        )
        p2 = ScoredPoint(
            id="2",
            version=1,
            score=0.79,
            payload={"tags": ["Gesundheit"]},
            vector=None,
        )
        p3 = ScoredPoint(
            id="3",
            version=1,
            score=0.75,
            payload={"tags": ["Sonstiges"]},
            vector=None,
        )
        return [p1, p2, p3]

    mock_qdrant.search_by_vector.side_effect = fake_search

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=mock_qdrant,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Arztbericht")

    assert "Gesundheit" in result.predicted_tags
    # "Sonstiges" only appeared once (< 2), so it is excluded
    assert "Sonstiges" not in result.predicted_tags


@pytest.mark.asyncio
async def test_knn_failure_resilience():
    """Engine gracefully continues classification when Qdrant search raises an error."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_ollama.get_document_embeddings.return_value = ([0.0] * 8, [0.0] * 8)

    mock_qdrant = AsyncMock(spec=QdrantStorageManager)
    mock_qdrant.search_by_vector.side_effect = QdrantStorageError("Qdrant unavailable")

    anchor_dict: dict[str, list[float]] = {}
    for a in get_tax_anchors():
        anchor_dict[a.name] = [0.0] * 8

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=mock_qdrant,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Dokument")
    assert result.is_tax_relevant is False
    assert result.tax_confidence == 0.0


# ---------------------------------------------------------------------------
# Anchor Caching and Pre-computation Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_anchor_initialization_caching():
    """initialize_anchor_embeddings embeds all anchors once and reuses them."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_ollama.get_embedding.return_value = [0.1] * 8
    mock_ollama.get_document_embeddings.return_value = ([0.1] * 8, [0.1] * 8)

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=None,
    )

    total_anchors = (
        len(get_document_type_anchors())
        + len(get_tax_anchors())
        + len(get_thematic_tag_anchors())
    )

    await engine.initialize_anchor_embeddings()
    assert mock_ollama.get_embedding.call_count == total_anchors

    # Reset call count
    mock_ollama.get_embedding.reset_mock()

    # Now classify: anchors should be read from cache, no get_embedding calls for anchors
    await engine.classify(content="Test")
    assert mock_ollama.get_embedding.call_count == 0


@pytest.mark.asyncio
async def test_engine_close():
    """Verify close cleanly shuts down both clients."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_qdrant = AsyncMock(spec=QdrantStorageManager)
    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=mock_qdrant,
    )
    await engine.close()
    mock_ollama.close.assert_awaited_once()
    mock_qdrant.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_knn_string_tag_payload():
    """Verify handling of scalar string tag in neighbor payload."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_ollama.get_document_embeddings.return_value = ([0.0] * 8, [0.0] * 8)

    mock_qdrant = AsyncMock(spec=QdrantStorageManager)
    p1 = ScoredPoint(
        id="1",
        version=1,
        score=0.8,
        payload={"tags": "Gesundheit"},
        vector=None,
    )
    p2 = ScoredPoint(
        id="2",
        version=1,
        score=0.79,
        payload={"tags_original": "Gesundheit"},
        vector=None,
    )
    mock_qdrant.search_by_vector.side_effect = lambda vector_name, **kwargs: (
        [] if kwargs.get("filter_criteria") else [p1, p2]
    )

    anchor_dict = {a.name: [0.0] * 8 for a in get_thematic_tag_anchors()}
    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=mock_qdrant,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Dokument")
    assert "Gesundheit" in result.predicted_tags
    assert result.tag_scores["Gesundheit"] == 1.0


@pytest.mark.asyncio
async def test_knn_tax_weighted_score_and_cluster_selection():
    """Verify weighted score calculation across top-5 neighbors and category cluster selection."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_ollama.get_document_embeddings.return_value = ([0.0] * 8, [0.0] * 8)

    # Baseline anchors return low similarity (0.40)
    anchor_dict = {
        a.name: make_unit_vector_with_similarity(0.40, dim=8) for a in get_tax_anchors()
    }

    # 3 neighbors: 2 in werbungskosten (0.90, 0.85) and 1 in sonderausgaben (0.80)
    # Sum of scores = 2.55, sum of squares = 0.81 + 0.7225 + 0.64 = 2.1725
    # Weighted score = round(2.1725 / 2.55, 4) = 0.8520
    # Cluster weights: werbungskosten = 1.75, sonderausgaben = 0.80
    p1 = ScoredPoint(
        id="1",
        version=1,
        score=0.90,
        payload={"is_tax_relevant": True, "tax_category": "steuer:werbungskosten"},
        vector=None,
    )
    p2 = ScoredPoint(
        id="2",
        version=1,
        score=0.85,
        payload={"is_tax_relevant": True, "tax_category": "steuer:werbungskosten"},
        vector=None,
    )
    p3 = ScoredPoint(
        id="3",
        version=1,
        score=0.80,
        payload={"is_tax_relevant": True, "tax_category": "steuer:sonderausgaben"},
        vector=None,
    )

    mock_qdrant = AsyncMock(spec=QdrantStorageManager)

    def mock_search(vector_name, **kwargs):
        crit = kwargs.get("filter_criteria", {})
        if crit.get("is_tax_relevant") is False:
            return []
        if crit.get("is_tax_relevant") is True or crit.get("is_tax_curated") is True:
            return [p1, p2, p3]
        return []

    mock_qdrant.search_by_vector.side_effect = mock_search

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=mock_qdrant,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Notebook Rechnung")

    assert result.is_tax_relevant is True
    assert result.tax_review_required is False
    assert result.tax_confidence == 0.8520
    assert result.tax_category == "steuer:werbungskosten"


@pytest.mark.asyncio
async def test_knn_canonical_macro_tag_mapping():
    """Verify raw tags from neighbor payloads map to canonical macro-tags before voting."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_ollama.get_document_embeddings.return_value = ([0.0] * 8, [0.0] * 8)

    anchor_dict = {a.name: [0.0] * 8 for a in get_thematic_tag_anchors()}

    # p1 has "kfz" (maps to Fahrzeug) and an unmapped tag
    # p2 has "auto" (also maps to Fahrzeug)
    # p3 has "strom" (maps to Energie, 1 vote only)
    # p4 and p5 have unmapped custom tags
    p1 = ScoredPoint(
        id="1",
        version=1,
        score=0.8,
        payload={"tags": ["kfz", "custom-unmapped"]},
        vector=None,
    )
    p2 = ScoredPoint(
        id="2", version=1, score=0.8, payload={"tags": ["auto"]}, vector=None
    )
    p3 = ScoredPoint(
        id="3", version=1, score=0.8, payload={"tags": ["strom"]}, vector=None
    )
    p4 = ScoredPoint(
        id="4", version=1, score=0.8, payload={"tags": ["random-tag"]}, vector=None
    )
    p5 = ScoredPoint(
        id="5", version=1, score=0.8, payload={"tags": ["Sonstiges"]}, vector=None
    )

    mock_qdrant = AsyncMock(spec=QdrantStorageManager)
    mock_qdrant.search_by_vector.side_effect = lambda vector_name, **kwargs: (
        [] if kwargs.get("filter_criteria") else [p1, p2, p3, p4, p5]
    )

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=mock_qdrant,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Kfz Inspektion")

    # "Fahrzeug" has 2 neighbor votes (>= 2) -> assigned
    assert "Fahrzeug" in result.predicted_tags
    # "Energie" has only 1 neighbor vote (< 2) -> excluded
    assert "Energie" not in result.predicted_tags
    # Unmapped tags and "Sonstiges" -> excluded
    assert "custom-unmapped" not in result.predicted_tags
    assert "Sonstiges" not in result.predicted_tags


# ---------------------------------------------------------------------------
# Contrastive Tax k-NN Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_contrastive_knn_excludes_when_nontax_neighbor_wins():
    """Verify document is excluded when non-tax neighbor clearly beats tax score (>= +0.03)."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_ollama.get_document_embeddings.return_value = ([0.0] * 8, [0.0] * 8)

    anchor_dict = {a.name: [0.0] * 8 for a in get_tax_anchors()}

    tax_pt = ScoredPoint(
        id="1",
        version=1,
        score=0.86,
        payload={"is_tax_relevant": True, "tax_category": "steuer:werbungskosten"},
        vector=None,
    )
    nontax_pt = ScoredPoint(
        id="2",
        version=1,
        score=0.96,
        payload={"is_tax_relevant": False},
        vector=None,
    )

    mock_qdrant = AsyncMock(spec=QdrantStorageManager)

    def mock_search_by_vector(vector_name, **kwargs):
        crit = kwargs.get("filter_criteria", {})
        if crit.get("is_tax_relevant") is False:
            return [nontax_pt]
        if crit.get("is_tax_relevant") is True:
            return [tax_pt]
        return []

    mock_qdrant.search_by_vector.side_effect = mock_search_by_vector

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=mock_qdrant,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Wertpapierabrechnung Kauf Aktien")

    # 0.96 (NonTax) >= 0.86 (Tax) + 0.03 -> Excluded
    assert result.is_tax_relevant is False
    assert result.tax_review_required is False
    assert result.tax_category is None


@pytest.mark.asyncio
async def test_contrastive_knn_marks_review_on_close_boundary():
    """Verify document is marked for review when tax score is high and close to non-tax score."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_ollama.get_document_embeddings.return_value = ([0.0] * 8, [0.0] * 8)

    anchor_dict = {a.name: [0.0] * 8 for a in get_tax_anchors()}

    tax_pt = ScoredPoint(
        id="1",
        version=1,
        score=0.88,
        payload={"is_tax_relevant": True, "tax_category": "steuer:werbungskosten"},
        vector=None,
    )
    nontax_pt = ScoredPoint(
        id="2",
        version=1,
        score=0.89,
        payload={"is_tax_relevant": False},
        vector=None,
    )

    mock_qdrant = AsyncMock(spec=QdrantStorageManager)

    def mock_search_by_vector(vector_name, **kwargs):
        crit = kwargs.get("filter_criteria", {})
        if crit.get("is_tax_relevant") is False:
            return [nontax_pt]
        if crit.get("is_tax_relevant") is True:
            return [tax_pt]
        return []

    mock_qdrant.search_by_vector.side_effect = mock_search_by_vector

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=mock_qdrant,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Grenzfall Dokument")

    # NonTax (0.89) - Tax (0.88) = 0.01 < 0.03 -> Boundary case queued for review
    assert result.is_tax_relevant is True
    assert result.tax_review_required is True
    assert result.tax_category == "steuer:werbungskosten"


@pytest.mark.asyncio
async def test_contrastive_knn_confident_tax_when_tax_neighbor_wins():
    """Verify document is confident tax when tax neighbor clearly beats non-tax neighbor."""
    mock_ollama = AsyncMock(spec=OllamaEmbeddingClient)
    mock_ollama.get_document_embeddings.return_value = ([0.0] * 8, [0.0] * 8)

    anchor_dict = {a.name: [0.0] * 8 for a in get_tax_anchors()}

    tax_pt = ScoredPoint(
        id="1",
        version=1,
        score=0.95,
        payload={"is_tax_relevant": True, "tax_category": "steuer:handwerker"},
        vector=None,
    )
    nontax_pt = ScoredPoint(
        id="2",
        version=1,
        score=0.65,
        payload={"is_tax_relevant": False},
        vector=None,
    )

    mock_qdrant = AsyncMock(spec=QdrantStorageManager)

    def mock_search_by_vector(vector_name, **kwargs):
        crit = kwargs.get("filter_criteria", {})
        if crit.get("is_tax_relevant") is False:
            return [nontax_pt]
        if crit.get("is_tax_relevant") is True:
            return [tax_pt]
        return []

    mock_qdrant.search_by_vector.side_effect = mock_search_by_vector

    engine = VectorClassifierEngine(
        ollama_client=mock_ollama,
        qdrant_manager=mock_qdrant,
        anchor_embeddings=anchor_dict,
    )
    result = await engine.classify(content="Handwerkerrechnung Malerarbeiten")

    assert result.is_tax_relevant is True
    assert result.tax_review_required is False
    assert result.tax_category == "steuer:handwerker"
    assert result.tax_confidence == 0.95
