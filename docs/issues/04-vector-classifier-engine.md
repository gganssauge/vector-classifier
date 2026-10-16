# 04: Vektor-Klassifikations-Engine (Dokumenttyp & Hybride Steuerrelevanz)

**Status:** completed
**Blocked by:** 01-ollama-model-pull-and-client, 02-semantic-anchors-catalogue, 03-qdrant-collection-and-schema

## What to build

1. **Klassifikations-Engine (`src/vector_classifier/vectors/classifier.py`):**
   - Bindeglied zwischen Ollama-Embeddings, semantischen Ankern und Qdrant-Nachbarschaften.
2. **Dokumenttyp-Klassifikation:**
   - Berechnet Ähnlichkeit des `header`-Vektors gegen die 12 vorab eingebetteten Dokumenttyp-Anker.
   - Ermittelt Top-1-Treffer mit Cosine-Score.
   - Bestimmt Konfidenzlevel (`confident` bei $\ge 0{,}80$, `ambiguous` bei geringem Abstand zwischen Platz 1 und 2).
3. **Hybride Steuerrelevanz-Klassifikation:**
   - Zweistufige Bewertung des `content`-Vektors:
     - **Stufe A (Semantische Anker):** Cosine-Similarity gegen die 5 steuerlichen Abzugskategorien (Werbungskosten, § 35a, Sonderausgaben, Krankheit, Anlage V).
     - **Stufe B (k-NN zu kuratierten Vorjahresbelegen):** Vektorsuche in Qdrant mit Filter `is_tax_relevant: true`. Ermittelt gewichteten Score der Top-5-Nachbarn.
   - **Kombinierter Steuer-Score:** $Score_{tax} = \max(Score_{anchor}, Score_{knn})$.
   - **Drei-Zonen-Auswertung:**
     - $Score_{tax} \ge 0{,}82 \rightarrow$ `steuerrelevant` (und spezifische `tax_category`).
     - $0{,}70 \le Score_{tax} < 0{,}82 \rightarrow$ `steuer-review`.
     - $Score_{tax} < 0{,}70 \rightarrow$ nicht steuerrelevant.
4. **Multi-Label Thematisches Tagging:**
   - Berechnet Ähnlichkeit des `content`-Vektors gegen die 14 thematischen Tag-Anker.
   - k-NN Tag-Aggregation aus den Top-5 Qdrant-Nachbarn mit Mapping auf kanonische Makro-Tags.
   - Weist alle Tags zu, die den Schwellenwert $\ge 0{,}78$ erreichen oder von mindestens 2 Nachbarn unterstützt werden.
5. **Ergebnis-Schema (`VectorClassificationResult`):**
   - Typisierte Pydantic-Klasse:
     - `predicted_doc_type: str`
     - `doc_type_confidence: float`
     - `predicted_tags: list[str]`
     - `tag_scores: dict[str, float]`
     - `is_tax_relevant: bool`
     - `tax_confidence: float`
     - `tax_category: str | None`
     - `tax_review_required: bool`
     - `matched_anchors: list[dict[str, Any]]`

## Acceptance Criteria

- [x] Dokumenttypen werden mit Ähnlichkeitswerten aus `header`-Vektoren gegen Anker bestimmt.
- [x] Steuerrelevanz kombiniert semantische Steuer-Anker und k-NN-Vorjahresnachbarn.
- [x] Schwellenwerte für `steuerrelevant` ($\ge 0{,}82$) und `steuer-review` ($0{,}70\text{--}0{,}81$) werden exakt eingehalten.
- [x] Multi-Label Tagging weist alle qualifizierten Makro-Tags ($\ge 0{,}78$) sauber als Liste zu.
- [x] Vollständige Unit-Tests mit synthetischen Vektoren verifizieren die mathematische Zuordnung und Randfälle.
