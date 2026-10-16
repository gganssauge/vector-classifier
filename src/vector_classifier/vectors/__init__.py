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

__all__ = [
    "OllamaClientError",
    "OllamaConnectionError",
    "OllamaEmbeddingClient",
    "OllamaResponseError",
    "get_document_type_anchors",
    "get_tax_anchors",
    "get_thematic_tag_anchors",
]
