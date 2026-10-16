# Vector Classifier (`vector-classifier`)

> Standalone Microservice & Batch Indexer for Semantic Vector Classification, Tax Relevance Assessment, and Similarity Search powered by **Ollama (`bge-m3`)** and **Qdrant**.

[![CI Status](https://img.shields.io/badge/CI-passing-brightgreen)](#)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Architecture: Decoupled](https://img.shields.io/badge/architecture-decoupled-success)](#)

---

## Overview

`vector-classifier` is a lightweight, asynchronous microservice designed to perform dense semantic document classification without local GPU dependencies. It delegates vector embeddings via HTTP to an **Ollama** instance running the multilingual `bge-m3` model on AMD ROCm, and persists named document vectors into a **Qdrant** vector database.

### Key Capabilities

- **Zero-Shot Document Type Classification:** Matches document headers (`header` vector, ~1.000 tokens) against 12 canonical semantic type anchors via Cosine Similarity.
- **Hybrid Tax Relevance Assessment:** Combines semantic anchors for German tax deduction categories (§ 35a Handwerkerleistungen, Werbungskosten, Sonderausgaben, außergewöhnliche Belastungen, Anlage V) with k-NN similarity to curated historical receipts.
- **Three-Zone Confidence Model:**
  - $\ge 0{,}82 \rightarrow$ High confidence: automatically marked as tax-relevant with specific deduction category.
  - $0{,}70\text{--}0{,}81 \rightarrow$ Medium confidence: marked for manual human review (`steuer-review`).
  - $< 0{,}70 \rightarrow$ Low confidence: no action.
- **Multi-Label Thematic Tagging:** Classifies documents into 14 consolidated macro-categories (e.g., `Fahrzeug`, `Energie`, `Telekommunikation`, `Gesundheit`, `Versicherung`, `Immobilien`).
- **Deterministic & Idempotent:** Points in Qdrant use a deterministic UUIDv5 derived from the document's content SHA-256 hash.
- **Dual Mode (API & CLI):** Serves real-time REST requests for orchestrators (e.g., `paperless-orchestrator`) and offers a CLI for batch backfills and offline dry-run audit passes.

---

## Architecture

```text
 ┌────────────────────────┐
 │ paperless-orchestrator │ (or other workflow orchestrators)
 └───────────┬────────────┘
             │ POST /api/v1/classify
             ▼
 ┌────────────────────────────────────────────────────────┐
 │                   vector-classifier                    │
 │                                                        │
 │  1. Extract header & content text chunks               │
 │  2. Generate embeddings via Ollama (bge-m3)            │
 │  3. Search & compare vectors against anchors in Qdrant │
 │  4. Compute hybrid tax score & macro-tags              │
 └───────────┬────────────────────────────────┬───────────┘
             │                                │
             ▼ HTTP (async)                   ▼ HTTP (async)
    ┌─────────────────┐             ┌────────────────────┐
    │     Ollama      │             │       Qdrant       │
    │ (bge-m3 / ROCm) │             │ (paperless_docs)   │
    └─────────────────┘             └────────────────────┘
```

---

## Configuration

Settings are managed via environment variables or a `.env` file using `pydantic-settings`.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://ollama.ai.svc.cluster.local:11434` | URL of the Ollama service hosting `bge-m3`. |
| `OLLAMA_EMBED_MODEL` | `bge-m3` | Embedding model identifier in Ollama. |
| `OLLAMA_TIMEOUT` | `60.0` | Timeout in seconds for embedding generation. |
| `QDRANT_HOST` | `qdrant.ai.svc.cluster.local` | Hostname of the Qdrant vector database. |
| `QDRANT_PORT` | `6333` | Port of the Qdrant REST/gRPC API. |
| `QDRANT_COLLECTION` | `paperless_documents` | Collection name for vector storage. |
| `QDRANT_API_KEY` | `""` | Optional API key for Qdrant authentication. |
| `HOST` | `0.0.0.0` | Bind host for FastAPI server. |
| `PORT` | `8000` | Bind port for FastAPI server. |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `THRESHOLD_TAX_CONFIDENT` | `0.82` | Cosine similarity threshold for automatic tax tagging. |
| `THRESHOLD_TAX_REVIEW` | `0.70` | Lower threshold for tax review flagging. |
| `THRESHOLD_TAG` | `0.78` | Cosine similarity threshold for macro-tag assignment. |
| `THRESHOLD_DOCTYPE_CONFIDENT` | `0.80` | Minimum score for confident document type detection. |

---

## REST API Endpoints

### `GET /health`

Liveness and readiness check verifying connectivity to Ollama and Qdrant.

### `POST /api/v1/classify`

Takes document text and returns predicted document type, tax relevance evaluation, and thematic tags.

**Request:**

```json
{
  "content": "Rechnung Nr. 2026-4421...",
  "filename": "rechnung_telekom.pdf",
  "sha256": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a"
}
```

**Response:**

```json
{
  "predicted_doc_type": "Rechnung",
  "doc_type_confidence": 0.91,
  "is_tax_relevant": true,
  "tax_confidence": 0.86,
  "tax_category": "werbungskosten",
  "tax_review_required": false,
  "predicted_tags": ["Telekommunikation"],
  "tag_scores": {
    "Telekommunikation": 0.89
  }
}
```

---

## CLI Usage

The CLI tool `vector-classifier` provides commands for batch operations, collection schema initialization, and offline audit passes:

```bash
# Initialize or verify Qdrant collection schema
uv run vector-classifier ensure-collection

# Run an offline dry-run audit pass
uv run vector-classifier audit --limit 50 --output-csv audit_results.csv
```

---

## Development & Testing

### Virtual Environment Setup

```bash
# Install dependencies with uv
uv sync
```

### Running Tests

```bash
# Run pytest with asyncio and mock HTTP fixtures
uv run pytest

# Run with test coverage
uv run pytest --cov=vector_classifier --cov-report=term-missing
```

### Code Formatting & Quality

Code style is strictly enforced via `ruff` and `ty` in git pre-commit hooks (`prek`).

---

## Documentation

- [Technical Specification](docs/spec.md) — Comprehensive specification of vectors, collections, and scoring algorithms.
- [Implementation Tickets](docs/issues/) — Modular work packages (01 to 06).
- [Agent Guidelines](AGENTS.md) — Rules, commands, and conventions for coding agents.

---

## License

This project is licensed under the MIT License.
