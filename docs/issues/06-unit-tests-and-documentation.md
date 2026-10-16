# 06: Testsuite-Härtung, Regressionstests & Dokumentations-Sync

**Status:** ready-for-agent
**Blocked by:** 05-batch-indexer-and-audit-runner

## What to build

1. **Unit-Test-Suite (`tests/test_vector_classifier.py`):**
   - Mocks für `OllamaEmbeddingClient` (liefert deterministische Test-Vektoren ohne echten Ollama-Server).
   - In-Memory- oder Mock-Tests für `QdrantStorageManager` (Simulation von Upsert und Kosinus-Ähnlichkeitssuche).
   - Testabdeckung für Edge Cases:
     - Leere OCR-Texte oder fehlende Titel.
     - Sehr lange Texte (Verhalten beim Splitten von Header vs. Content).
     - Grenzwert-Tests für die Drei-Zonen-Logik ($0{,}69$, $0{,}70$, $0{,}81$, $0{,}82$).
2. **Dokumentations-Synchronisation:**
   - Aktualisierung von `docs/project-structure.md` mit den neuen Modulpfaden unter `custom_classifier/vectors/`.
   - Dokumentation der Architektur und Qdrant-Schnittstelle in `docs/agents/domain.md`.
   - Eintrag in `README.md` zur optionalen Aktivierung der Vektorklassifikation.

## Acceptance Criteria

- [ ] Alle neuen Tests laufen grün über `uv run pytest tests/test_vector_*.py`.
- [ ] Gesamte Testsuite `uv run pytest` bleibt fehlerfrei und ohne Regressionen.
- [ ] Dokumentation entspricht den Richtlinien aus `documentation-policy.md`.
- [ ] Keine SonarCloud-/Linter-Warnungen (`ruff`, Komplexität, Typehints).
