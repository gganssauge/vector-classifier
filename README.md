# Vector Classifier (`vector-classifier`)

> Standalone Microservice & Batch Indexer for Semantic Vector Classification, Tax Relevance Assessment, and Similarity Search powered by **Ollama (`bge-m3`)** and **Qdrant**.

[![CI Status](https://img.shields.io/badge/CI-passing-brightgreen)](#)
[![Python 3.15+](https://img.shields.io/badge/python-3.15%2B-blue.svg)](https://www.python.org/downloads/)
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

### `ingest`

Ingests documents from Paperless-ngx directly into Qdrant, computing `header` and `content` dense vectors asynchronously via Ollama (`bge-m3`):

```bash
# Ingest all new/updated documents from Paperless
uv run vector-classifier ingest

# Recreate Qdrant collection fresh and re-ingest all documents
uv run vector-classifier ingest --recreate --concurrency 8 --batch-size 50
```

### `calibrate-tax`

Calibrates tax relevance across three temporal zones using contrastive k-NN (`Score(Tax) vs Score(NonTax)`):

```bash
# Run 3-zone tax relevance calibration and export 2026+ candidates
uv run vector-classifier calibrate-tax

# Calibrate with custom threshold and files
uv run vector-classifier calibrate-tax --threshold-knn 0.82 --candidates-csv tax_candidates_2026.csv
```

### `tag-candidates`

Tags all tax candidates in Paperless in seconds using the Paperless Bulk API (`POST /api/documents/bulk_edit/`):

```bash
# Dry-run: inspect what would be tagged without modifying Paperless
uv run vector-classifier tag-candidates --dry-run

# Apply tags in Paperless (Steuern-Review + proposed Steuern-Privat / Steuern-Vermietung)
uv run vector-classifier tag-candidates
```

### `sync-curated`

Synchronizes reviewed candidate documents from Paperless back into Qdrant and updates ground-truth files:

```bash
# Dry-run check: verify status of human review
uv run vector-classifier sync-curated --dry-run

# Apply sync: marks confirmed tax docs and records confirmed non-tax anchors
uv run vector-classifier sync-curated
```

---

## Tax Relevance & Active Learning Workflow

When new documents arrive in Paperless throughout the year, or after preparing the tax return, use this recurring 5-step active learning cycle:

```text
 ┌────────────────┐     ┌───────────────────┐     ┌───────────────────┐
 │ 1. Ingest      │ ──> │ 2. Calibrate      │ ──> │ 3. Bulk Tag       │
 │    (Qdrant DB) │     │    (Find & Score) │     │    (Paperless UI) │
 └────────────────┘     └───────────────────┘     └─────────┬─────────┘
                                                            │
 ┌────────────────┐     ┌───────────────────┐               │
 │ 5. Verified!   │ <── │ 4. Sync Curated   │ <─────────────┘
 │    (0 pending) │     │    (Active Learn) │   Human Review in UI
 └────────────────┘     └───────────────────┘   (Remove 'Steuern-Review')
```

### Step-by-Step Guide

1. **Ingest New Documents:**
   Index any newly added or scanned documents from Paperless into Qdrant:

   ```bash
   uv run vector-classifier ingest
   ```

2. **Calibrate Tax Relevance:**
   Run contrastive k-NN against curated tax anchors and non-tax negative examples:

   ```bash
   uv run vector-classifier calibrate-tax
   ```

   This generates `tax_candidates_2026.csv` with newly discovered tax candidates (excluding bank statements and known non-tax patterns).

3. **Tag Candidates in Paperless:**
   Apply tags via the Paperless Bulk API:

   ```bash
   uv run vector-classifier tag-candidates
   ```

   Each candidate receives the yellow tag `Steuern-Review` **plus** the proposed category tag (`Steuern-Privat` or `Steuern-Vermietung`).

4. **Human Review in Paperless UI:**
   Filter Paperless documents by the tag `Steuern-Review`:
   - **Accept / Confirmed:** Delete **only** the `Steuern-Review` tag. The proposed category tag (`Steuern-Privat` or `Steuern-Vermietung`) stays on the document (1 click).
   - **Reject / Not Tax-Relevant:** Delete **both** tags (or remove all tax tags).

5. **Synchronize & Learn (Active Learning):**
   Sync the human review decisions back into Qdrant and the curated datasets:

   ```bash
   uv run vector-classifier sync-curated
   ```

   - Confirmed tax receipts are marked `is_tax_curated=True` and appended to `steuern-privat.txt` or `steuern-vermietung.txt`.
   - Rejected receipts (e.g. personal cable TV, securities settlements) are stored as curated negative examples (`tax_source="confirmed_nontax"`), preventing similar future documents from triggering false positives.

6. **Verify Calibration:**
   Re-run calibration to confirm all candidates are settled:

   ```bash
   uv run vector-classifier calibrate-tax
   ```

   Expected output: `Zone 2026+ (Candidates): 0 candidates`.

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
- [Domain Modeling](docs/agents/domain.md) — Deep dive into named vectors, semantic anchors, three-zone tax logic, and UUIDv5 indexing.
- [Implementation Tickets](docs/issues/) — Modular work packages (01 to 06).
- [Agent Guidelines](AGENTS.md) — Rules, commands, and conventions for coding agents.

---

## License

This project is licensed under the MIT License.
