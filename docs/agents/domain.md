# Domain Concepts & Modeling

This document describes the core domain concepts, semantic structures, vector spaces, and decision rules implemented in `vector-classifier`.

---

## 1. Named Vectors Architecture

`vector-classifier` uses the multilingual `bge-m3` embedding model (1,024 vector dimensions, 8,192 token context window). Rather than embedding an entire document into a single vector, documents are partitioned into two dedicated **Named Vectors** in Qdrant:

```text
Document Text
 ├── Header Section (~3,000 chars / ~1,000 tokens) ──► Named Vector "header" (1024d, Cosine)
 └── Full Content Text (up to 8,192 tokens)       ──► Named Vector "content" (1024d, Cosine)
```

### `header` Vector

- **Source:** Extracted from the document's first ~1,000 tokens (up to ~3,000 characters), trimmed neatly at paragraph or line boundaries.
- **Purpose:** Document type identification (e.g., `Rechnung`, `Vertrag`, `Kontoauszug`). Briefköpfe, logos, sender information, and title lines are concentrated in the first page and define document typologies without distraction from lengthy tabular bodies.
- **Metric:** Cosine distance.

### `content` Vector

- **Source:** The full document body text (up to 8,192 tokens).
- **Purpose:** Broad semantic classification, tax deductibility assessment, and multi-label thematic topic categorization.
- **Metric:** Cosine distance.

---

## 2. Semantic Anchors

Semantic anchors represent canonical domain concepts embedded into dense vector space at service startup. The catalog defines three categories:

### Document Types (12 Anchors)

Zero-shot anchors evaluated against the `header` vector:

| Anchor | Description & Keywords |
| :--- | :--- |
| `Rechnung` | Rechnungen über Warenlieferungen oder Dienstleistungen mit Zahlungsziel. |
| `Quittung` | Zahlungsbelege, Kassenbons, Barzahlungsquittungen. |
| `Abrechnung` | Gehaltsabrechnungen, Lohnabrechnungen, Nebenkostenabrechnungen. |
| `Kontoauszug` | Bankkontoauszüge, Kreditkartenabrechnungen, Finanzübersichten. |
| `Vertrag` | Verträge, Vereinbarungen, Mietverträge, Arbeitsverträge. |
| `Bescheid` | Behördliche Bescheide, Steuerbescheide, Verwaltungsakte. |
| `Versicherungsschein` | Versicherungspolicen, Versicherungsscheine, Deckungszusagen. |
| `Mahnung` | Zahlungserinnerungen, Mahnungen, Inkassoforderungen. |
| `Bestätigung` | Anmeldebestätigungen, Buchungsbestätigungen, Immatrikulationsbescheinigungen. |
| `Zertifikat` | Urkunden, Zeugnisse, Bescheinigungen, Prüfungszertifikate. |
| `Arztbrief` | Medizinische Befunde, Arztbriefe, Laborberichte, Rezepte. |
| `Brief` | Allgemeine Korrespondenz, Anschreiben, Informationsschreiben. |

### Tax Deduction Categories (5 Anchors)

Evaluated against the `content` vector:

| Anchor | Deduction Category | Legal Context |
| :--- | :--- | :--- |
| `steuer:haushaltsnah-35a` | Handwerkerleistungen & haushaltsnahe Dienstleistungen | § 35a EStG |
| `steuer:werbungskosten` | Arbeitsmittel, Fachliteratur, Fortbildung, Bewerbungen, Dienstreisen | § 9 EStG |
| `steuer:sonderausgaben` | Spenden, Vorsorgeaufwendungen, Kirchensteuer, Kinderbetreuung | §§ 10, 10b EStG |
| `steuer:krankheit-aussergewoehnlich` | Krankheitskosten, Zahnersatz, Medikamente, Pflegekosten | § 33 EStG |
| `steuer:vermietung-anlage-v` | Instandhaltung, Grundsteuer, Hausverwaltung, Hausgeld | Anlage V EStG |

### Thematic Macro-Tags (14 Anchors)

Evaluated against the `content` vector for multi-label assignment:

- `Energie` (Strom, Gas, Fernwärme, Heizkosten)
- `Fahrzeug` (Kfz-Kauf, Werkstatt, TÜV, Leasing)
- `Finanzen` (Bankgebühren, Kredite, Zinsen, Depot)
- `Gesundheit` (Arzt, Apotheke, Behandlungen, Krankenkasse)
- `Haushalt` (Haushaltsgeräte, Möbel, Gartenpflege)
- `Immobilien` (Mietvertrag, Nebenkosten, Hausverwaltung)
- `IT & Elektronik` (Software, Hardware, Cloud, Domain)
- `Reise` (Flug, Hotel, Bahn, Mietwagen)
- `Steuern` (Finanzamt, Steuerberater, Elster)
- `Telekommunikation` (Mobilfunk, Internet, Telefon, Festnetz)
- `Unterhaltung` (Streaming, Musik, Gaming, Bücher)
- `Versicherung` (Haftpflicht, Hausrat, Berufsunfähigkeit, Kfz-Versicherung)
- `Verträge` (Laufzeitverträge, Kündigungen, Mitgliedschaften)
- `Wohnen` (Möbel, Renovierung, Dekoration)

---

## 3. Three-Zone Tax Model

Tax relevance follows a three-zone decision policy combining cosine similarity to tax anchors with k-NN similarity to historical receipts:

```text
Similarity Score
 1.00 ┌──────────────────────────────────────────────┐
      │  Zone 1: Confident                           │  is_tax_relevant = True
 0.82 ├──────────────────────────────────────────────┤  tax_review_required = False
      │  Zone 2: Review                              │  is_tax_relevant = True
 0.70 ├──────────────────────────────────────────────┤  tax_review_required = True
      │  Zone 3: No Tax Relevance                    │  is_tax_relevant = False
 0.00 └──────────────────────────────────────────────┘  tax_review_required = False
```

### Decision Rules

1. **Confident Zone ($\ge 0{,}82$):**
   - Automatically flagged as `is_tax_relevant = True`.
   - `tax_review_required = False`.
   - `tax_category` is assigned from the top-scoring tax anchor or high-confidence k-NN match.
2. **Review Zone ($0{,}70 \le \text{score} < 0{,}82$):**
   - Flagged as `is_tax_relevant = True`.
   - `tax_review_required = True` (routes document to human review queue in Paperless).
   - `tax_category` is suggested for verification.
3. **Ignored Zone ($< 0{,}70$):**
   - `is_tax_relevant = False`.
   - `tax_review_required = False`.
   - `tax_category = None`.

### k-NN Synergy

When Qdrant contains indexed documents with confirmed tax annotations:

- A top-scoring k-NN neighbor with `is_tax_relevant = True` can elevate the overall tax score above the baseline anchor score.
- If the nearest neighbor's score exceeds the anchor score, its verified `tax_category` is inherited.

---

## 4. Multi-Label Thematic Macro-Tags

Thematic tagging operates in multi-label mode:

1. **Anchor Thresholding:**
   Any macro-tag whose anchor similarity is $\ge 0{,}78$ is included in `predicted_tags`.
2. **k-NN Voting:**
   Qdrant is queried for the 5 nearest neighbors in the `content` vector space. Any tag appearing in $\ge 2$ neighbors is assigned, even if its direct semantic anchor score is below $0{,}78$.

---

## 5. Idempotent Point IDs (UUIDv5)

Document points in Qdrant use deterministic UUIDv5 identifiers:

$$\text{Point ID} = \text{UUIDv5}(\text{NAMESPACE\_DNS}, \text{sha256}.\text{strip}().\text{lower}())$$

### Guarantees

- **Idempotence:** Indexing the same document text multiple times produces the exact same point ID, preventing duplicate points.
- **Content-Addressable:** Changing the content hash triggers a new point, ensuring vector consistency.
- **Paperless Decoupled:** Does not rely on transient database sequence IDs from upstream systems.
