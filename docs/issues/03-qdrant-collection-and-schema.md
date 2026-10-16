# 03: Qdrant Collection Layout & Idempotentes Schema-Management

**Status:** completed
**Blocked by:** 01-ollama-model-pull-and-client

## What to build

1. **Qdrant Storage Manager (`src/vector_classifier/vectors/qdrant.py`):**
   - Asynchroner Client für Qdrant (unter Nutzung von `qdrant-client` oder direktem REST-Client).
   - Konfigurierbare Endpoints über `QDRANT_HOST` (Default: `qdrant.ai.svc.cluster.local`), `QDRANT_PORT` (Default: `6333`), `QDRANT_COLLECTION` (Default: `paperless_documents`).
2. **Idempotente Collection-Initialisierung (`ensure_collection`):**
   - Prüft, ob Collection existiert. Falls nicht, Neuanlage mit Named Vectors:
     - `header`: `VectorParams(size=1024, distance=Distance.COSINE)`
     - `content`: `VectorParams(size=1024, distance=Distance.COSINE)`
   - Erstellung von Payload-Indexen für schnelle Filterung:
     - `sha256` (Keyword)
     - `paperless_id` (Integer)
     - `doc_type` (Keyword)
     - `is_tax_relevant` (Boolean)
     - `tax_category` (Keyword)
3. **Deterministische ID-Generierung:**
   - Funktion `generate_point_id(content_hash: str) -> str`: Erzeugt reproduzierbare UUIDv5 (Namespace DNS/URL) aus dem SHA-256-Hash des Dokuments.
4. **Upsert- und Query-Schnittstelle:**
   - `upsert_document(...)`: Schreibt oder aktualisiert Dokument-Punkte idempotent.
   - `search_by_vector(...)`: Sucht ähnliche Dokumente nach Named Vector mit optionalem Payload-Filter.

## Acceptance Criteria

- [x] Collection `paperless_documents` wird mit zwei 1024-dimensionalen Cosine-Vektoren angelegt.
- [x] Point-IDs basieren deterministisch auf dem Content-Hash (keine zufälligen UUIDs).
- [x] Upsert ist idempotent (erneutes Ausführen überschreibt identische Punkte ohne Duplikate).
- [x] Payload-Filterung nach `is_tax_relevant`, `paperless_id` und `doc_type` funktioniert performant.
- [x] Unit-Tests mit Mock-Qdrant prüfen Initialisierung und Upsert-Payloads.
