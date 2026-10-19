# Checkliste Vektor-DB

## Steuern

[x] Groundtruth

[x] manueller Abgleich

## Dokumenttypen

[x] Groundtruth

[x] manueller Abgleich

## Tags

[x] Groundtruth

[x] manueller Abgleich

## Integration in automatische Qualifikation

### Batchdurchlauf (Bestandsqualifikation)

[x] Dry-Run & Validierung des Gesamtbestands (4.688 Dokumente)

[x] Bulk-Applikation in Paperless (`uv run vector-classifier apply-batch --no-dry-run`)

### Event-Pipeline (Orchestrator-Integration)

[x] REST-API Endpoint (`POST /api/v1/classify`) für Orchestrator

[ ] Anbindung an Orchestrator & custom-classifier (Titel, Korrespondent, Datum, LLM)
