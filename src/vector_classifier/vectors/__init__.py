"""Vector operations and anchors module."""

from vector_classifier.vectors.anchors import (
    get_document_type_anchors,
    get_tax_anchors,
    get_thematic_tag_anchors,
)
from vector_classifier.vectors.client import (
    OllamaClientError,
    OllamaConnectionError,
    OllamaEmbeddingClient,
    OllamaResponseError,
)
from vector_classifier.vectors.qdrant import (
    QdrantConnectionError,
    QdrantStorageError,
    QdrantStorageManager,
    generate_point_id,
)

__all__ = [
    "OllamaClientError",
    "OllamaConnectionError",
    "OllamaEmbeddingClient",
    "OllamaResponseError",
    "QdrantConnectionError",
    "QdrantStorageError",
    "QdrantStorageManager",
    "generate_point_id",
    "get_document_type_anchors",
    "get_tax_anchors",
    "get_thematic_tag_anchors",
]
