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

- **Named Vectors Architecture:** Splitting documents into two distinct 1,024-dimensional dense vectors:
  - `header`: Briefkopf / first ~1,000 tokens (~3,000 characters) for zero-shot document type classification.
  - `content`: Fließtext (up to 8,192 tokens) for tax relevance assessment and thematic macro-tags.
- **Zero-Shot Document Type Classification:** Matches document headers (`header` vector) against 12 canonical semantic type anchors via Cosine Similarity.
- **Hybrid Tax Relevance Assessment:** Combines semantic anchors for German tax deduction categories (§ 35a Handwerkerleistungen, Werbungskosten, Sonderausgaben, außergewöhnliche Belastungen, Anlage V) with k-NN similarity against curated historical receipts.
- **Three-Zone Confidence Model:**
  - $\ge 0{,}82 \rightarrow$ High confidence: automatically marked as tax-relevant with specific deduction category.
  - $0{,}70\text{--}0{,}81 \rightarrow$ Medium confidence: marked for manual human review (`steuer-review`).
  - $< 0{,}70 \rightarrow$ Low confidence: no action taken.
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

Liveness and readiness check verifying service status and connectivity to Ollama and Qdrant.

**Response:**

```json
{
  "status": "ok",
  "version": "0.1.0",
  "ollama_connected": true,
  "qdrant_connected": true
}
```

### `POST /api/v1/classify`

Classifies document text and returns predicted document type, tax relevance assessment, and thematic macro-tags.

**Request:**

```json
{
  "content": "Rechnung Nr. 2026-4421\nTelekom Deutschland GmbH\nMonatliche Abrechnung Internet & Festnetz\nGesamtbetrag: 49,95 EUR",
  "filename": "rechnung_telekom.pdf",
  "sha256": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
  "header_content": null
}
```

**Response:**

```json
{
  "predicted_doc_type": "Rechnung",
  "doc_type_confidence": 0.91,
  "is_tax_relevant": true,
  "tax_confidence": 0.86,
  "tax_category": "steuer:werbungskosten",
  "tax_review_required": false,
  "predicted_tags": [
    "Telekommunikation"
  ],
  "tag_scores": {
    "Telekommunikation": 0.89
  },
  "matched_anchors": [
    {
      "name": "Rechnung",
      "category": "doc_type",
      "similarity": 0.91,
      "confident": true,
      "ambiguous": false
    },
    {
      "name": "steuer:werbungskosten",
      "category": "tax",
      "similarity": 0.86,
      "knn_tax_score": 0.0
    },
    {
      "name": "Telekommunikation",
      "category": "theme",
      "similarity": 0.89
    }
  ]
}
```

---

## CLI Usage

The service provides a Typer CLI via `uv run vector-classifier` (or `python -m vector_classifier.cli`).

### `info`

Displays the current active configuration, Ollama/Qdrant connection endpoints, and classification thresholds:

```bash
uv run vector-classifier info
```

### `version`

Displays the package version:

```bash
uv run vector-classifier version
```

### `run`

Starts the FastAPI server with `uvicorn`:

```bash
# Start server on default host and port (0.0.0.0:8000)
uv run vector-classifier run

# Start server with live reload and custom port
uv run vector-classifier run --host 127.0.0.1 --port 8080 --reload
```

### `audit`

Executes batch vector classification and offline evaluation against exported Paperless document dumps (JSON or JSONL):

```bash
# Dry-run audit on documents without writing changes
uv run vector-classifier audit --source documents.jsonl --output-csv audit_report.csv

# Process first 50 documents with concurrency limit of 8
uv run vector-classifier audit --source documents.json --limit 50 --concurrency 8

# Index vectorized documents into Qdrant during audit
uv run vector-classifier audit --source documents.jsonl --index --output-csv audit_report.csv
```

**Options:**

- `--source`, `-s`: Path to input JSON (list of document objects) or JSONL file (*required*).
- `--output-csv`, `-o`: Destination path for tabular CSV evaluation metrics (default: `audit_report.csv`).
- `--limit`, `-l`: Maximum number of documents to evaluate.
- `--dry-run` / `--no-dry-run`: Dry-run mode without modifying Paperless (default: `--dry-run`).
- `--index` / `--no-index`: Whether to upsert vectorized documents into Qdrant (default: `--no-index`).
- `--concurrency`, `-c`: Maximum concurrent embedding requests to Ollama (default: 4).

### Standalone Audit Script

For convenience, `scripts/vector_audit_pass.py` provides a direct entry point:

```bash
uv run scripts/vector_audit_pass.py --source export.jsonl --output-csv audit.csv
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
# Run full test suite with pytest
uv run pytest

# Run with coverage report
uv run pytest --cov=vector_classifier --cov-report=term-missing
```

### Code Formatting & Quality

Code style and types are checked automatically via pre-commit hooks (`prek`) on `git commit`:

- `ruff check` and `ruff format`
- `ty check`
- `markdownlint-cli2`

---

## Documentation

- [Technical Specification](docs/spec.md) — Comprehensive specification of vectors, collections, and scoring algorithms.
- [Project Structure](docs/project-structure.md) — Codebase architecture and module map.
- [Domain Modeling](docs/domain.md) — Deep dive into named vectors, semantic anchors, three-zone tax logic, and UUIDv5 indexing.
- [Implementation Tickets](docs/issues/) — Modular work packages (01 to 06).
- [Agent Guidelines](AGENTS.md) — Rules, commands, and conventions for coding agents.

---

## License

This project is licensed under the MIT License.
