# Spezifikation: Qdrant Vektor-Klassifikation mit BGE-M3 (Dokumenttypen & Steuerrelevanz)

Status: ready-for-agent

## Problem Statement

Paperless-ngx verfügt über einen stetig wachsenden Bestand an Dokumenten, deren OCR-Qualität durch moderne OCR-Pipelines (z. B. PaddleOCR / ROCm) signifikant verbessert wurde. Bisherige Klassifikationsansätze basieren entweder auf statischen Regex-Mustern oder flachen TF-IDF-Modellen (Scikit-Learn). Diese Ansätze stoßen an klare Grenzen:

1. **Begrenztes semantisches Verständnis:** Belege für Arbeitsmittel, Fachliteratur, Werkzeuge oder Handwerkerleistungen (§ 35a EStG) enthalten selten das Signalwort „Steuer“. Reine Volltext- und Keyword-Regeln übersehen solche Belege systematisch oder erfordern ausufernde, fehleranfällige Regex-Kataloge.
2. **Abhängigkeit von Paperless-IDs und starrem Schema:** Bestehende Klassifikatoren sind eng an Paperless-interne IDs gekoppelt. Wird Paperless neu aufgesetzt oder werden Dokumente außerhalb verarbeitet, gehen Verknüpfungen verloren.
3. **Dokumenttyp-Unschärfe:** Dokumenttypen wie „Rechnung“, „Abrechnung“, „Bescheid“ oder „Bescheinigung“ sind semantisch nuanciert. Reine Volltext-Matches scheitern an layout-spezifischen Feinheiten (z. B. Arztrechnungen vs. Beitragsrechnungen vs. Erstattungsmitteilungen).
4. **Verfügbare Cluster-Ressourcen ungenutzt:** Im Kubernetes-Cluster laufen im Namespace `ai` bereits eine **Qdrant-Vektordatenbank** (`v1.19.2`) und ein GPU-beschleunigtes **Ollama** (`rocm`). Diese Infrastruktur soll erschlossen werden, um dichte semantische Vektorrepräsentationen (Dense Embeddings) für Klassifikation und Recherche nutzbar zu machen.

---

## Solution

Ein entkoppelter Vektorisierungs- und Klassifikations-Pass, der OCR-Texte mittels des multilingualen Embedding-Modells `bge-m3` über Ollama in Qdrant überführt und für Zero-Shot/Few-Shot-Klassifikation nutzt:

1. **Modell & Inferenz (`bge-m3` via Ollama):**
   - Nutzung von `bge-m3` (8.192 Tokens Kontextlänge, 1.024 Dimensionen, herausragende Leistung bei deutschem Verwaltungs-, Rechts- und Steuerwortschatz).
   - Inferenz läuft hardwarebeschleunigt auf der AMD-GPU im Cluster über den Ollama-Service (`http://ollama.ai.svc.cluster.local:11434`).
2. **Qdrant-Architektur mit Named Vectors:**
   - Collection `paperless_documents` mit zwei Named Vectors:
     - `header`: Erste ca. 1.000 Tokens (Briefkopf, Betreff, Aussteller) für die trennscharfe Bestimmung von **Dokumenttyp** und Korrespondent.
     - `content`: Gesamtdokument (bis 8.192 Tokens) für **Thementags** und **Steuerrelevanz**.
3. **Deterministische Entkopplung:**
   - Punkt-ID in Qdrant ist eine deterministische UUIDv5, abgeleitet aus dem **SHA-256-Hash** des Inhalts.
   - Paperless-Metadaten (`paperless_id`, `filename`, `created`, `tags`) werden als reine Payload-Attribute geführt.
4. **Semantische Anker (Zero-Shot) für Dokumenttypen:**
   - Für die 12 kanonischen Paperless-Dokumenttypen werden detaillierte, semantisch reiche deutsche Beschreibungen als Vektoren hinterlegt.
   - Die Zuordnung erfolgt über Cosine-Similarity des `header`-Vektors gegen die Anker-Vektoren.
5. **Hybride Steuerrelevanz-Erkennung:**
   - Abgleich des `content`-Vektors gegen semantische Steuer-Anker (Werbungskosten, Handwerker § 35a, Sonderausgaben, Krankheitskosten, Anlage V Vermietung).
   - Ergänzender k-NN-Abgleich gegen die kuratierten Vorjahresbelege (2019–2026).
   - Drei-Zonen-Modell: Hohe Konfidenz ($\ge 0{,}82$) $\rightarrow$ `steuerrelevant`, mittlere Konfidenz ($0{,}70\text{--}0{,}81$) $\rightarrow$ `steuer-review`, darunter keine Aktion.
6. **Thematische Tag-Taxonomie & Multi-Label Tagging:**
   - Konsolidierung historischer Tags in ca. 12–15 kanonische Makro-Gruppen (z. B. `Fahrzeug`, `Energie`, `Telekommunikation`, `Gesundheit`, `Versicherung`, `Immobilien`).
   - Jede Gruppe besitzt einen semantischen Vektor-Anker mit Kontextbegriffen und Synonymen.
   - Multi-Label Inferenz: Neues Dokument prüft `content`-Vektor gegen alle Tag-Anker; alle Gruppen mit Ähnlichkeit $\ge 0{,}78$ (oder Mehrheitsvotum der Top-5 Bestandsnachbarn) werden als Tags gesetzt.
7. **Audit-Pass & Trockenlauf:**
   - Ein dediziertes CLI-Skript (`--dry-run`) importiert Dokumente in Qdrant, evaluiert die Trefferquote und erzeugt einen tabellarischen CSV/Markdown-Audit-Report ohne Schreibzugriff auf Paperless.

---

## User Stories

1. Als Benutzer möchte ich, dass `bge-m3` als Embedding-Modell in Ollama auf der AMD-GPU ausgeführt wird, damit selbst mehrseitige Dokumente (bis 8.192 Tokens) schnell und ohne Informationsverlust vektorisiert werden.
2. Als Benutzer möchte ich, dass jedes Dokument in Qdrant über eine deterministische UUID aus seinem Content-SHA-256 identifiziert wird, damit Vektoren idempotent und unabhängig von Paperless-IDs bleiben.
3. Als Benutzer möchte ich zwei Named Vectors (`header` und `content`) in Qdrant haben, damit Briefkopf-Merkmale (Dokumenttyp) nicht durch lange Fließtexte verwässert werden.
4. Als Benutzer möchte ich für alle 12 kanonischen Dokumenttypen semantisch reichhaltige deutsche Beschreibungen als Referenzanker pflegen, damit Dokumenttypen transparent und anpassbar über Cosine-Similarity erkannt werden.
5. Als Benutzer möchte ich, dass thematische Tags aus einer sauberen Makro-Taxonomie (z. B. `Fahrzeug`, `Energie`, `Telekommunikation`) als Multi-Label automatisch vergeben werden, sobald ein Dokument Ähnlichkeit mit den Themen-Ankern oder Bestands-Clustern aufweist.
6. Als Benutzer möchte ich, dass steuerrelevante Dokumente (Arbeitsmittel, § 35a Handwerker, Krankheitskosten, Hausgeld/Vermietung) auch ohne explizite Steuerschlagwörter zuverlässig erkannt werden.
7. Als Benutzer möchte ich ein Drei-Zonen-Konfidenzmodell für Steuern, damit sichere Treffer direkt markiert werden und Grenzfälle als `steuer-review` zur Prüfung landen.
8. Als Benutzer möchte ich einen Offline-Audit-Pass ausführen können (`--dry-run`), der mir eine detaillierte Auswertung (vorgeschlagener Dokumenttyp, vorgeschlagene Tags, Steuerrelevanz, Konfidenzwerte) als CSV liefert, ohne den Live-Bestand in Paperless zu verändern.
9. Als Entwickler möchte ich robuste Unit-Tests mit Mocks für Ollama und Qdrant, damit die gesamte Vektorisierungs- und Klassifikationslogik in CI ohne externe Abhängigkeiten getestet werden kann.

---

## Implementation Decisions

### 1. Inferenz- und Netzwerk-Architektur

- **Ollama API:** Verwendung des Standard-Endpoints `/api/embeddings` von Ollama.
- **Client:** Asynchroner HTTP-Client (`httpx`) mit Retry-Logik, Timeout (30s) und Batching (z. B. 10 Dokumente parallel).
- **Fallback für lokale Ausführung:** Erkennung, ob Cluster-DNS (`ollama.ai.svc.cluster.local:11434`) oder ein lokaler Port-Forward (`localhost:11434`) genutzt wird über Umgebungsvariable `OLLAMA_BASE_URL`.

### 2. Qdrant Collection Layout

- **Collection-Name:** `paperless_documents`
- **Vectors Configuration:**
  - `header`: `size=1024`, `distance=Cosine`
  - `content`: `size=1024`, `distance=Cosine`
- **Optimierung:** On-Disk Payload mit Payload-Indexen auf `sha256`, `is_tax_relevant`, `doc_type`, `paperless_id`.

### 3. Semantischer Anker-Katalog (`custom_classifier/vectors/anchors.py`)

- Python-Modul mit typisierten Datenstrukturen für Dokumenttyp- und Steuer-Definitionen.
- 12 Dokumenttypen: *Rechnung, Abrechnung, Korrespondenz, Kontoauszug, Bescheinigung, Bescheid, Vertrag, Steuern, Gesundheit, Urkunde, Sonstiges* mit mehrsätzigen, präzisen Kontextbeschreibungen.
- 5 Steuerkategorien: *Werbungskosten (Arbeitsmittel, IT, Literatur), Handwerker (§ 35a EStG), Sonderausgaben & Vorsorge, Außergewöhnliche Belastungen (Krankheitskosten), Anlage V (Vermietung, Hausgeld, Instandhaltung)*.

### 4. Entkopplung & Payload-Struktur

```json
{
  "id": "uuid5-from-sha256",
  "payload": {
    "sha256": "abcdef...",
    "filename": "scan_2026.pdf",
    "paperless_id": 4812,
    "source_system": "paperless",
    "title": "Telekom Rechnung März 2026",
    "doc_type_original": "Rechnung",
    "tags_original": ["Telekommunikation"],
    "is_tax_relevant": true,
    "tax_category": "werbungskosten",
    "indexed_at": "2026-10-13T20:30:00Z"
  }
}
```

### 5. Drei-Zonen-Schwellenwerte & Scoring

- Ähnlichkeitsscore $S \in [-1, 1]$ (Cosine Similarity):
  - $S \ge 0{,}82$: Hohe Konfidenz (`confident`)
  - $0{,}70 \le S < 0{,}82$: Mittlere Konfidenz (`review`)
  - $S < 0{,}70$: Niedrige Konfidenz (`unassigned` / keine Aktion)
