"""Vector classification engine for document types, hybrid tax relevance, and thematic tags."""

import asyncio
import logging
import math
from typing import Any

from vector_classifier.config import settings
from vector_classifier.models.schemas import SemanticAnchor, VectorClassificationResult
from vector_classifier.vectors.anchors import (
    get_document_type_anchors,
    get_tax_anchors,
    get_thematic_tag_anchors,
)
from vector_classifier.vectors.client import OllamaEmbeddingClient
from vector_classifier.vectors.qdrant import QdrantStorageManager

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two float vectors.

    Returns float in [-1.0, 1.0]. Returns 0.0 if vectors are empty,
    have mismatched lengths, or have zero norm.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    val = dot_product / (norm1 * norm2)
    return max(-1.0, min(1.0, val))


def extract_header_content(content: str, max_chars: int = 3000) -> str:
    """Extract header content (first ~1000 tokens or ~3000 characters)."""
    if not content:
        return ""
    if len(content) <= max_chars:
        return content
    cut_idx = content.rfind("\n", 0, max_chars)
    if cut_idx > 500:
        return content[:cut_idx].strip()
    return content[:max_chars].strip()


def extract_body_content(content: str, max_chars: int = 8000) -> str:
    """Extract content body safely bounded by context window (up to ~8000 characters)."""
    if not content:
        return ""
    if len(content) <= max_chars:
        return content
    cut_idx = content.rfind("\n", 0, max_chars)
    if cut_idx > 2000:
        return content[:cut_idx].strip()
    return content[:max_chars].strip()


def _anchor_to_embedding_text(anchor: SemanticAnchor) -> str:
    """Format anchor metadata into text for dense embedding."""
    text = f"{anchor.name}: {anchor.description}"
    if anchor.keywords:
        text += f" ({', '.join(anchor.keywords)})"
    return text


def map_to_canonical_macro_tag(raw_tag: str) -> str | None:
    """Map raw tag string to canonical macro-tag matching anchor name or keywords."""
    if not raw_tag or not isinstance(raw_tag, str):
        return None
    normalized = raw_tag.strip().lower()
    for anchor in get_thematic_tag_anchors():
        if normalized == anchor.name.lower():
            return anchor.name
        if any(normalized == kw.lower() for kw in anchor.keywords):
            return anchor.name
    return None


_DEFAULT: Any = object()


class VectorClassifierEngine:
    """Orchestration engine for vector-based zero-shot and few-shot classification.

    Coordinates Ollama embedding generation, semantic anchor matching,
    and k-NN proximity evaluation in Qdrant.
    """

    def __init__(
        self,
        ollama_client: OllamaEmbeddingClient | None = None,
        qdrant_manager: Any = _DEFAULT,
        anchor_embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        self.ollama_client = (
            ollama_client if ollama_client is not None else OllamaEmbeddingClient()
        )
        self.qdrant_manager: QdrantStorageManager | None = (
            QdrantStorageManager() if qdrant_manager is _DEFAULT else qdrant_manager
        )
        self._anchor_embeddings: dict[str, list[float]] = (
            dict(anchor_embeddings) if anchor_embeddings is not None else {}
        )
        self._anchors_initialized = bool(anchor_embeddings)

    def get_anchor_embedding(self, category: str, name: str) -> list[float] | None:
        """Retrieve pre-computed anchor embedding from cache."""
        return self._anchor_embeddings.get(
            f"{category}:{name}"
        ) or self._anchor_embeddings.get(name)

    async def _get_or_compute_anchor_embedding(
        self, anchor: SemanticAnchor
    ) -> list[float]:
        """Retrieve anchor embedding from cache or embed on-demand."""
        cached = self.get_anchor_embedding(anchor.category, anchor.name)
        if cached is not None:
            return cached
        text = _anchor_to_embedding_text(anchor)
        vec = await self.ollama_client.get_embedding(text)
        self._anchor_embeddings[f"{anchor.category}:{anchor.name}"] = vec
        self._anchor_embeddings[anchor.name] = vec
        return vec

    async def initialize_anchor_embeddings(self) -> None:
        """Pre-compute and cache embeddings for all canonical semantic anchors."""
        all_anchors = (
            get_document_type_anchors() + get_tax_anchors() + get_thematic_tag_anchors()
        )
        to_embed = [
            anchor
            for anchor in all_anchors
            if self.get_anchor_embedding(anchor.category, anchor.name) is None
        ]
        if to_embed:
            tasks = [
                self.ollama_client.get_embedding(_anchor_to_embedding_text(a))
                for a in to_embed
            ]
            embeddings = await asyncio.gather(*tasks)
            for anchor, emb in zip(to_embed, embeddings):
                self._anchor_embeddings[f"{anchor.category}:{anchor.name}"] = emb
                self._anchor_embeddings[anchor.name] = emb
        self._anchors_initialized = True

    async def close(self) -> None:
        """Close underlying client connections."""
        if self.ollama_client is not None:
            await self.ollama_client.close()
        if self.qdrant_manager is not None:
            await self.qdrant_manager.close()

    async def _classify_doc_type(
        self, header_vector: list[float]
    ) -> tuple[str | None, float, list[dict[str, Any]]]:
        """Classify document type using header vector against semantic anchors."""
        doc_type_anchors = get_document_type_anchors()
        doc_type_scores: list[tuple[SemanticAnchor, float]] = []
        for anchor in doc_type_anchors:
            anchor_vec = await self._get_or_compute_anchor_embedding(anchor)
            sim = cosine_similarity(header_vector, anchor_vec)
            doc_type_scores.append((anchor, sim))

        doc_type_scores.sort(key=lambda x: x[1], reverse=True)
        if not doc_type_scores:
            return None, 0.0, []

        top_anchor, top_score_raw = doc_type_scores[0]
        top_score = round(top_score_raw, 4)
        second_score = (
            round(doc_type_scores[1][1], 4) if len(doc_type_scores) > 1 else None
        )

        is_confident = top_score >= settings.threshold_doctype_confident
        is_ambiguous = (
            (top_score - second_score < 0.05) if second_score is not None else False
        )
        predicted_doc_type = top_anchor.name if is_confident else None

        matched_anchors = [
            {
                "name": top_anchor.name,
                "category": "doc_type",
                "similarity": top_score,
                "confident": is_confident,
                "ambiguous": is_ambiguous,
            }
        ]
        return predicted_doc_type, top_score, matched_anchors

    async def _classify_tax_relevance(
        self, content_vector: list[float]
    ) -> tuple[bool, float, str | None, bool, list[dict[str, Any]]]:
        """Classify hybrid tax relevance using semantic anchors and Qdrant k-NN search."""
        tax_anchors = get_tax_anchors()
        tax_scores: list[tuple[SemanticAnchor, float]] = []
        for anchor in tax_anchors:
            anchor_vec = await self._get_or_compute_anchor_embedding(anchor)
            sim = cosine_similarity(content_vector, anchor_vec)
            tax_scores.append((anchor, sim))

        tax_scores.sort(key=lambda x: x[1], reverse=True)
        best_tax_anchor, anchor_tax_score_raw = tax_scores[0]
        anchor_tax_score = round(anchor_tax_score_raw, 4)
        best_category: str | None = best_tax_anchor.name

        knn_tax_score, knn_category = await self._query_tax_knn(content_vector)
        tax_confidence = round(max(anchor_tax_score, knn_tax_score), 4)
        if knn_tax_score > anchor_tax_score and knn_category:
            best_category = knn_category

        if tax_confidence >= settings.threshold_tax_confident:
            is_tax_relevant = True
            tax_review_required = False
            tax_category = best_category
        elif tax_confidence >= settings.threshold_tax_review:
            is_tax_relevant = True
            tax_review_required = True
            tax_category = best_category
        else:
            is_tax_relevant = False
            tax_review_required = False
            tax_category = None

        matched_anchors = [
            {
                "name": best_tax_anchor.name,
                "category": "tax",
                "similarity": anchor_tax_score,
                "knn_tax_score": knn_tax_score,
            }
        ]
        return (
            is_tax_relevant,
            tax_confidence,
            tax_category,
            tax_review_required,
            matched_anchors,
        )

    async def _query_tax_knn(
        self, content_vector: list[float]
    ) -> tuple[float, str | None]:
        """Query Qdrant for tax-relevant nearest neighbors with weighted scoring."""
        if self.qdrant_manager is None:
            return 0.0, None
        try:
            tax_neighbors = await self.qdrant_manager.search_by_vector(
                vector_name="content",
                vector=content_vector,
                limit=5,
                filter_criteria={"is_tax_curated": True},
            )
            if not tax_neighbors:
                return 0.0, None

            # Calculate similarity-weighted score across top-5 tax neighbors
            scores = [max(0.0, float(getattr(n, "score", 0.0))) for n in tax_neighbors]
            total_score = sum(scores)
            if total_score <= 0.0:
                return 0.0, None

            # Similarity-weighted mean: sum(s_i^2) / sum(s_i)
            weighted_score = round(sum(s * s for s in scores) / total_score, 4)

            # Select category from highest-weighted category cluster
            cluster_weights: dict[str, float] = {}
            for n in tax_neighbors:
                score = max(0.0, float(getattr(n, "score", 0.0)))
                payload = getattr(n, "payload", {}) or {}
                cat = payload.get("tax_category")
                if cat and isinstance(cat, str):
                    cluster_weights[cat] = cluster_weights.get(cat, 0.0) + score

            best_category = (
                max(cluster_weights.items(), key=lambda item: item[1])[0]
                if cluster_weights
                else None
            )
            return weighted_score, best_category
        except Exception as exc:
            logger.warning("k-NN tax vector search failed: %s", exc)
        return 0.0, None

    async def _classify_thematic_tags(
        self, content_vector: list[float]
    ) -> tuple[list[str], dict[str, float], list[dict[str, Any]]]:
        """Classify thematic tags using semantic anchors and Qdrant k-NN voting."""
        tag_anchors = get_thematic_tag_anchors()
        tag_scores: dict[str, float] = {}
        qualifying_tags: set[str] = set()
        matched_anchors: list[dict[str, Any]] = []

        for anchor in tag_anchors:
            anchor_vec = await self._get_or_compute_anchor_embedding(anchor)
            sim = round(cosine_similarity(content_vector, anchor_vec), 4)
            tag_scores[anchor.name] = sim
            if sim >= settings.threshold_tag:
                qualifying_tags.add(anchor.name)
                matched_anchors.append(
                    {
                        "name": anchor.name,
                        "category": "theme",
                        "similarity": sim,
                    }
                )

        await self._apply_knn_tag_votes(content_vector, qualifying_tags, tag_scores)
        return sorted(list(qualifying_tags)), tag_scores, matched_anchors

    async def _apply_knn_tag_votes(
        self,
        content_vector: list[float],
        qualifying_tags: set[str],
        tag_scores: dict[str, float],
    ) -> None:
        """Query Qdrant neighbors and add canonical macro-tags supported by >= 2 neighbors."""
        if self.qdrant_manager is None:
            return
        try:
            all_neighbors = await self.qdrant_manager.search_by_vector(
                vector_name="content",
                vector=content_vector,
                limit=5,
            )
            if not all_neighbors:
                return

            neighbor_tag_counts: dict[str, int] = {}
            for n in all_neighbors:
                payload = getattr(n, "payload", {}) or {}
                raw_tags = payload.get("tags") or payload.get("tags_original") or []
                if isinstance(raw_tags, str):
                    raw_tags = [raw_tags]
                elif not isinstance(raw_tags, list):
                    raw_tags = []

                # Map raw tags to canonical macro-tags and deduplicate per neighbor
                neighbor_canonical_tags: set[str] = set()
                for t in raw_tags:
                    if isinstance(t, str):
                        canonical = map_to_canonical_macro_tag(t)
                        if canonical:
                            neighbor_canonical_tags.add(canonical)

                for tag in neighbor_canonical_tags:
                    neighbor_tag_counts[tag] = neighbor_tag_counts.get(tag, 0) + 1

            for tag, count in neighbor_tag_counts.items():
                if count >= 2:
                    qualifying_tags.add(tag)
                    vote_score = round(count / len(all_neighbors), 4)
                    tag_scores[tag] = max(tag_scores.get(tag, 0.0), vote_score)
        except Exception as exc:
            logger.warning("k-NN thematic tag search failed: %s", exc)

    async def classify(
        self,
        content: str,
        filename: str = "",
        sha256: str = "",
        header_content: str | None = None,
    ) -> VectorClassificationResult:
        """Execute full semantic vector classification pipeline for a document."""
        header_text = (
            header_content
            if header_content is not None
            else extract_header_content(content)
        )
        (
            header_vector,
            content_vector,
        ) = await self.ollama_client.get_document_embeddings(header_text, content)

        pred_doc_type, doc_confidence, doc_anchors = await self._classify_doc_type(
            header_vector
        )
        (
            is_tax,
            tax_conf,
            tax_cat,
            tax_review,
            tax_anchors,
        ) = await self._classify_tax_relevance(content_vector)
        pred_tags, tag_scores, theme_anchors = await self._classify_thematic_tags(
            content_vector
        )

        return VectorClassificationResult(
            predicted_doc_type=pred_doc_type,
            doc_type_confidence=doc_confidence,
            predicted_tags=pred_tags,
            tag_scores=tag_scores,
            is_tax_relevant=is_tax,
            tax_confidence=tax_conf,
            tax_category=tax_cat,
            tax_review_required=tax_review,
            matched_anchors=doc_anchors + tax_anchors + theme_anchors,
            header_vector=header_vector,
            content_vector=content_vector,
        )
