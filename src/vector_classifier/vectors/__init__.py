"""Vector operations, anchors, and classification engine module."""

from vector_classifier.models.schemas import VectorClassificationResult
from vector_classifier.vectors.anchors import (
    get_document_type_anchors,
    get_tax_anchors,
    get_thematic_tag_anchors,
)
from vector_classifier.vectors.audit import (
    AUDIT_CSV_COLUMNS,
    AuditRecord,
    AuditRunner,
    AuditSummary,
    run_audit,
)
from vector_classifier.vectors.classifier import (
    VectorClassifierEngine,
    cosine_similarity,
    extract_header_content,
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
    "AUDIT_CSV_COLUMNS",
    "AuditRecord",
    "AuditRunner",
    "AuditSummary",
    "OllamaClientError",
    "OllamaConnectionError",
    "OllamaEmbeddingClient",
    "OllamaResponseError",
    "QdrantConnectionError",
    "QdrantStorageError",
    "QdrantStorageManager",
    "VectorClassificationResult",
    "VectorClassifierEngine",
    "cosine_similarity",
    "extract_header_content",
    "generate_point_id",
    "get_document_type_anchors",
    "get_tax_anchors",
    "get_thematic_tag_anchors",
    "run_audit",
]
