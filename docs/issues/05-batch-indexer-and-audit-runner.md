# 05: Batch-Indexer & Offline-Audit-Runner CLI

**Status:** ready-for-agent
**Blocked by:** 04-vector-classifier-engine

## What to build

1. **CLI-Runner (`scripts/vector_audit_pass.py`):**
   - Eigenständiges Werkzeug zum Vektorisieren von Bestandsdokumenten und Evaluieren der Klassifikationsergebnisse.
   - Argumente:
     - `--dry-run`: Standardmodus, indexiert in Qdrant (oder isolierte Test-Collection), führt KEINE Schreiboperationen gegen Paperless aus.
     - `--limit <int>`: Begrenzung auf $N$ Dokumente für schnelle Validierungstests.
     - `--source <path_or_api>`: Unterstützung für lokale JSON/JSONL-Caches (`.scratch/data/...`) oder direkte Paperless-API.
     - `--output-csv <path>`: Pfad für die tabellarische Auswertungsdatei.
     - `--concurrency <int>`: Parallele Embedding-Requests an Ollama (Standard: 4, um GPU nicht zu fluten).
2. **Audit-Reporting:**
   - Erzeugt strukturierte CSV-Datei mit folgenden Spalten:
     - `paperless_id`, `filename`, `content_sha256`
     - `current_doc_type`, `predicted_doc_type`, `doc_type_confidence`, `doc_type_match` (True/False)
     - `current_tags`, `predicted_tags`, `tag_confidence_scores`
     - `is_currently_tax_tagged`, `predicted_tax_relevant`, `tax_confidence`, `tax_category`, `tax_action` (`auto-tag` / `review` / `none`)
   - Konsolen-Zusammenfassung:
     - Genauigkeit der Dokumenttyp-Zuordnung (Accuracy / Top-1 Match-Rate gegenüber Altdaten).
     - Anzahl zugewiesener thematischer Makro-Tags.
     - Anzahl neu entdeckter steuerrelevanter Belege.
     - Verteilung nach Konfidenz-Zonen.

## Acceptance Criteria

- [ ] CLI-Skript kann mit `--dry-run` und `--limit 20` fehlerfrei ausgeführt werden.
- [ ] Drosselung/Concurrency verhindert Timeouts an der Ollama-Schnittstelle.
- [ ] CSV-Auditdatei wird mit allen relevanten Diagnose- und Vergleichsspalten erzeugt.
- [ ] Konsolenausgabe liefert verständliche statistische Kennzahlen zur Erkennungsqualität.
