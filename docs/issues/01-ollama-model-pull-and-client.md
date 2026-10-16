# 01: Ollama BGE-M3 Model Deployment & Asynchroner Embedding-Client

**Status:** completed
**Blocked by:** None (can start immediately)

## What to build

1. **Modell-Deployment in Ollama:**
   - Bereitstellung des Modells `bge-m3` in der bestehenden Ollama-Instanz im Namespace `ai` (`ollama pull bge-m3`).
2. **Asynchroner Ollama Embedding-Client (`custom_classifier/vectors/client.py`):**
   - Asynchroner HTTP-Client basierend auf `httpx.AsyncClient` zur Kommunikation mit dem Ollama `/api/embeddings`-Endpoint.
   - Konfiguration über Umgebungsvariablen: `OLLAMA_BASE_URL` (Default: `http://ollama.ai.svc.cluster.local:11434`), `OLLAMA_EMBED_MODEL` (Default: `bge-m3`), `OLLAMA_TIMEOUT` (Default: `60.0`).
   - Methoden:
     - `get_embedding(text: str) -> list[float]`: Erzeugt einen 1.024-dimensionalen Vektor für einen Text.
     - `get_document_embeddings(header_text: str, content_text: str) -> tuple[list[float], list[float]]`: Erzeugt parallel `header`- und `content`-Embeddings.
     - `health_check() -> bool`: Prüft Erreichbarkeit und Vorhandensein des Modells.
   - Robustheit: Verbindungs-Retries bei temporären Netzwerkfehlern, sauberes Exception-Handling, Truncation/Sanitization von Steuerzeichen.

## Acceptance Criteria

- [x] `bge-m3` ist in Ollama auf dem Cluster verfügbar (`ollama list` zeigt `bge-m3`).
- [x] `OllamaEmbeddingClient` ist in `src/vector_classifier/vectors/client.py` implementiert.
- [x] Methoden erzeugen konsistent 1.024-dimensionale Float-Vektoren.
- [x] Konfigurierbare Fallbacks für lokale Entwicklung (z. B. via `localhost:11434` oder K8s Port-Forward).
- [x] Unit-Tests mit Mock-Responses für Erfolgs- und Fehlerfälle laufen grün.
