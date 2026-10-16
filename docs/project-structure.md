# Project Structure

This document details the file organization, module boundaries, and architectural components of `vector-classifier`.

---

## Directory Overview

```text
vector-classifier/
├── src/
│   └── vector_classifier/
│       ├── __init__.py           # Package version definition
│       ├── config.py             # Pydantic Settings & environment variables
│       ├── main.py               # FastAPI application, lifecycle & routes
│       ├── cli.py                # Typer CLI application commands
│       ├── audit.py              # Batch classification & offline evaluation engine
│       ├── models/
│       │   ├── __init__.py
│       │   └── schemas.py        # Pydantic request/response & anchor schemas
│       └── vectors/
│           ├── __init__.py
│           ├── client.py         # Async Ollama HTTP client for bge-m3 embeddings
│           ├── qdrant.py         # Qdrant client, schema initialization & search
│           ├── anchors.py        # Canonical semantic anchors catalogue
│           └── classifier.py     # Classification engine, scoring & k-NN fusion
├── scripts/
│   └── vector_audit_pass.py      # Executable CLI wrapper for batch audit runs
├── tests/
│   ├── conftest.py               # Shared test fixtures & client mock factories
│   ├── test_anchors.py           # Unit tests for anchor catalogue structure
│   ├── test_api.py               # Integration tests for FastAPI endpoints
│   ├── test_audit.py             # Tests for batch audit runner & CSV reporting
│   ├── test_cli.py               # CLI invocation & argument parsing tests
│   ├── test_config.py            # Settings parsing & default threshold tests
│   ├── test_edge_cases.py        # Boundary tests, long texts, empty OCR, Qdrant faults
│   ├── test_qdrant.py            # Qdrant collection setup, indexing & search tests
│   ├── test_vector_classifier.py # Classification engine, cosine sim & k-NN tests
│   └── test_vector_client.py     # Ollama client mock tests & error handling
└── docs/
    ├── spec.md                   # Architectural specification
    ├── project-structure.md      # This file
    ├── domain.md                 # Domain models, scoring thresholds & vectors
    └── issues/                   # Modular implementation tickets (01 to 06)
```

---

## Core Modules Description

### `src/vector_classifier/`

- **`config.py`**:
  Central configuration managed via `pydantic-settings`. Reads environment variables (`OLLAMA_BASE_URL`, `QDRANT_HOST`, `THRESHOLD_TAX_CONFIDENT`, etc.) with strongly typed fallbacks and validations.

- **`main.py`**:
  FastAPI application entrypoint.
  - Manages lifespan events (startup anchor pre-computation and graceful shutdown of HTTP / Qdrant clients).
  - Exposes REST endpoints:
    - `GET /health`: Health probe reporting connectivity to Ollama and Qdrant.
    - `POST /api/v1/classify`: Synchronous vector classification endpoint for document text payloads.

- **`cli.py`**:
  Typer-based command-line interface:
  - `version`: Prints active version string.
  - `info`: Displays current configuration and active thresholds.
  - `run`: Launches the web server with `uvicorn`.
  - `audit`: Initiates batch evaluation of document dumps with optional Qdrant indexing.

- **`audit.py`**:
  Batch processing orchestrator (`AuditRunner`):
  - Ingests JSON (array of document objects) or JSONL (line-delimited JSON).
  - Normalizes document fields (`content`, `filename`, `created`, `sha256`, existing tags and document types).
  - Executes concurrent embedding and classification using an `asyncio.Semaphore`.
  - Computes precision/recall comparisons against existing human annotations.
  - Writes comprehensive tabular CSV evaluation reports.

### `src/vector_classifier/models/`

- **`schemas.py`**:
  Data validation and transfer schemas:
  - `SemanticAnchor`: Definition for category, canonical name, description, and keywords.
  - `ClassifyRequest`: Document ingestion request payload with optional explicit `header_content`.
  - `ClassifyResponse` / `VectorClassificationResult`: Prediction outcome including confidence scores, matched anchors, and dense vector embeddings.
  - `HealthResponse`: Status of backend connections.

### `src/vector_classifier/vectors/`

- **`client.py` (`OllamaEmbeddingClient`)**:
  Asynchronous HTTP client for the cluster's GPU-backed Ollama instance (`bge-m3`).
  - Implements retry with exponential backoff on transient HTTP status codes (429, 502, 503, 504).
  - Sanitizes text inputs by stripping non-printable control characters.
  - Fallback token embedding for empty or whitespace-only documents.
  - Concurrently retrieves `header` and `content` vector embeddings.

- **`qdrant.py` (`QdrantStorageManager`)**:
  Asynchronous interface to the Qdrant vector database:
  - `ensure_collection`: Idempotently creates the collection with two 1,024-dimensional named vectors (`header`, `content`) using Cosine metric.
  - Sets up payload indexes (`sha256`, `paperless_id`, `doc_type`, `is_tax_relevant`, `tax_category`).
  - `generate_point_id`: Computes deterministic UUIDv5 from content SHA-256 for idempotent upserts.
  - `search_by_vector`: Nearest-neighbor retrieval supporting both `query_points` and fallback `search` APIs.

- **`anchors.py`**:
  Curated domain knowledge catalogue:
  - 12 canonical document types (Rechnung, Quittung, Abrechnung, Vertrag, etc.).
  - 5 German tax deduction anchors (§ 35a, Werbungskosten, Sonderausgaben, außergewöhnliche Belastungen, Anlage V).
  - 14 thematic macro-tags for document tagging.

- **`classifier.py` (`VectorClassifierEngine`)**:
  Orchestrates zero-shot anchor comparison and few-shot k-NN evaluation:
  - Header extraction logic (`extract_header_content`) clipping up to ~3,000 characters cleanly on newline boundaries.
  - Pre-computes and caches anchor embeddings in memory during startup.
  - Calculates Cosine Similarity with boundary and zero-norm safeguards.
  - Three-zone tax decision logic with k-NN score boosting and category inheritance.
  - Multi-label thematic tag assignment with k-NN neighbor voting.

---

## Scripts

- **`scripts/vector_audit_pass.py`**:
  Standalone command-line executable wrapper around `vector_classifier.cli:app` allowing operators to invoke audit passes directly (`uv run scripts/vector_audit_pass.py`).
