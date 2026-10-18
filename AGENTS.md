# Agent Guidelines & Repository Map

Standalone microservice for semantic vector classification, tax relevance evaluation, and similarity search powered by Ollama (`bge-m3` on AMD ROCm) and Qdrant.

## Essential Core Rules

- **Package Manager & Execution:** Always use `uv`. Execute Python commands and scripts via `uv run <command>` (never bare `python3` or `pip`).
- **Command Shapes & No Ad-Hoc `-c` One-Liners:** Niemals ad-hoc `python -c "..."` Einzeiler ausführen! Dies bricht das Prefix-Matching der Terminal-Freigaben und zwingt den Benutzer zur unzumutbaren manuellen Einzelbestätigung jedes Kommandos. Stattdessen immer feste, wiederverwendbare Skripte unter `.scratch/` anlegen oder standardisierte CLI-Befehle (`uv run pytest ...`) nutzen.
- **Git Branching:** Direct pushes to `main` are strictly forbidden. Always use feature branches (`feature/<name>`) and conventional commits (`feat:`, `fix:`, `docs:`, `chore:`).
- **Quality Gates:** Pre-commit hooks (`prek`) enforce `ruff format`, `ruff check`, and type checks automatically on `git commit`. Do not run them manually before committing.
- **Architecture Boundaries:** This microservice acts as an inference and orchestration client for Ollama and Qdrant. It does NOT install local GPU drivers or heavy PyTorch runtimes; all embedding generation is offloaded via HTTP to the cluster's GPU-backed Ollama instance (`bge-m3`).
- **Paperless Bulk API:** Wann immer Änderungen (Tags, Korrespondenten, Dokumenttypen etc.) an mehreren Paperless-Dokumenten durchgeführt werden, muss zwingend das native Bulk-API (`POST /api/documents/bulk_edit/`) verwendet werden. Einzelne `PATCH`-Aufrufe pro Dokument lösen in Paperless/Django jeweils vollständige Model-Validierungen, Suchindex-Reindizierungen (Tantivy/Whoosh) und WebSocket-Broadcasts aus und führen zu extremen Latenzen (Sekunden statt Millisekunden).

## Key Commands

- Run all tests: `uv run pytest`
- Run single test module: `uv run pytest tests/test_<module>.py -k test_<func>`
- Run with coverage: `uv run pytest --cov=vector_classifier --cov-report=term-missing`
- Start local development server: `uv run uvicorn vector_classifier.main:app --reload --port 8000`
- CLI commands: `uv run vector-classifier --help`
- Pre-commit check (diagnosis only): `uv run prek run --all-files` (runs automatically on `git commit`)

## Architecture & Domain Model

1. **Embedding Pipeline:**
   - Model: Multilingual `bge-m3` (8,192 token context, 1,024 vector dimensions).
   - Ingestion splits documents into two Named Vectors:
     - `header`: Briefkopf / erste ~1.000 Tokens zur trennscharfen Bestimmung des Dokumenttyps.
     - `content`: Fließtext (bis 8.192 Tokens) zur Erkennung von Thementags und Steuerrelevanz.
2. **Qdrant Storage & Deduplication:**
   - Collection: `paperless_documents`.
   - Point-ID: Deterministische UUIDv5, abgeleitet aus dem Content-SHA-256 (vollständig idempotent).
3. **Klassifikations-Strategie:**
   - **Dokumenttypen (Zero-Shot):** Cosine-Similarity des `header`-Vektors gegen 12 kuratierte kanonische semantische Anker.
   - **Steuerrelevanz (Hybrid):** Abgleich des `content`-Vektors gegen 5 steuerliche Abzugskategorien (§ 35a, Werbungskosten, Sonderausgaben, Krankheit, Anlage V) kombiniert mit k-NN-Ähnlichkeit gegen Vorjahresbelege.
     - $\ge 0{,}82 \rightarrow$ `confident` (`steuerrelevant`)
     - $0{,}70\text{--}0{,}81 \rightarrow$ `review` (`steuer-review`)
     - $< 0{,}70 \rightarrow$ unberührt
   - **Thematische Makro-Tags:** Multi-Label Matching gegen 14 semantische Tag-Anker ($\ge 0{,}78$).

## Project Structure

- `src/vector_classifier/`
  - `config.py` — Typisierte Pydantic-Settings für Ollama, Qdrant und Schwellenwerte.
  - `main.py` — FastAPI Anwendungs-Einstiegspunkt.
  - `cli.py` — Typer CLI für Batch-Indexierung und Offline-Audits.
  - `models/` — Pydantic Schemas für API Requests/Responses.
  - `vectors/` — Ollama HTTP Client, Qdrant Manager, Anker-Katalog und Klassifikations-Engine.
- `docs/`
  - `spec.md` — Vollständige Architekturspezifikation.
  - `issues/` — Modulare Umsetzungs-Tickets (01 bis 06).
- `tests/` — Vollständige Unit- und Mock-Testsuite (respx).
