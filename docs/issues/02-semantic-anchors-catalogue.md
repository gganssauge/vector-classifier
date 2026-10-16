# 02: Semantischer Anker-Katalog für Dokumenttypen & Steuer-Kategorien

**Status:** ready-for-agent
**Blocked by:** None

## What to build

1. **Strukturierte Typdefinitionen (`custom_classifier/vectors/anchors.py`):**
   - `SemanticAnchor`: Datenklasse mit `name: str`, `category: str`, `description: str`, `keywords: list[str]`.
2. **Katalog der 12 kanonischen Dokumenttypen:**
   - Reichhaltige deutsche Beschreibungstexte für jeden Typ:
     - `Rechnung`: Kaufbelege, Rechnungsnummer, Leistungszeitraum, Zahlungsziel, Umsatzsteuer, IBAN, Einzelpositionen von Waren oder Dienstleistungen.
     - `Abrechnung`: Wiederkehrende Abrechnungen (Nebenkosten-, Betriebs- und Heizkostenabrechnung, Hausgeld, Lohn-/Gehaltsabrechnung, Kreditkarten- oder Zinsabrechnung).
     - `Kontoauszug`: Bankauszüge, Buchungstag, Wertstellung, Kontostand, Umsätze, Saldo.
     - `Bescheid`: Verwaltungsakte und behördliche Festsetzungen (Steuerbescheid, Gebührenbescheid, Grundsteuerbescheid, Bußgeldbescheid, Rentenbescheid).
     - `Bescheinigung`: Offizielle Bestätigungen ohne unmittelbare Zahlungsaufforderung (Spendenbescheinigung, Immatrikulationsbescheinigung, Arbeitsbescheinigung, Meldebescheinigung).
     - `Vertrag`: Rechtsgeschäftliche Vereinbarungen (Mietvertrag, Arbeitsvertrag, Kaufvertrag, Versicherungsschein, Police, AGB, Kündigung).
     - `Korrespondenz`: Allgemeiner Schriftverkehr, formlose Briefe, Mitteilungen, Anfragen ohne Rechnungs- oder Bescheidcharakter.
     - `Steuern`: Steuererklärungen, Elster-Übertragungsprotokolle, Mitteilungen des Finanzamts, Vorauszahlungsfestsetzungen.
     - `Gesundheit`: Medizinische Dokumente, Arztbriefe, Befunde, Rezepte, Patientenverfügungen, Entlassberichte.
     - `Urkunde`: Notarielle Urkunden, Geburts-, Heirats- und Sterbeurkunden, Grundbuchauszüge.
     - `Sonstiges`: Dokumente ohne eindeutige Zuordnung zu anderen Kernkategorien.
3. **Katalog der 5 steuerlichen Abzugskategorien:**
   - `steuer:werbungskosten`: Aufwendungen für den Beruf (Arbeitsmittel, PC/Hardware, Bürobedarf, Fachliteratur, Fachsoftware, Fortbildungen, Fahrtkosten).
   - `steuer:haushaltsnah-35a`: Handwerkerleistungen und haushaltsnahe Dienstleistungen nach § 35a EStG (Renovierungsarbeiten, Maler, Schornsteinfeger, Gartenpflege, Reparaturen, Reinigung).
   - `steuer:sonderausgaben`: Vorsorgeaufwendungen (Versicherungsbeiträge, Riester/Rürup), Spenden, Mitgliedsbeiträge, Kirchensteuer.
   - `steuer:krankheit-aussergewoehnlich`: Außergewöhnliche Belastungen (Arztrechnungen, Zahnbehandlungen, Medikamente, Zuzahlungen, Brillen, Hörgeräte, Heilpraktiker, Fahrtkosten zu Ärzten).
   - `steuer:anlage-v`: Einkünfte aus Vermietung und Verpachtung (Hausgeldabrechnungen, Gebäudeversicherung, Grundsteuer, Erhaltungsaufwand, Reparaturen an Mietobjekten).
4. **Katalog der thematischen Tag-Taxonomie (Makro-Tags):**
   - 14 kanonische Themenbereiche mit reichhaltiger Kontextbeschreibung und Synonym-Clustern:
     - `Fahrzeug` (Kfz, Auto, Werkstatt, Tanken, TÜV, Leasing, Reifen, Zulassung)
     - `Energie` (Strom, Gas, Fernwärme, Wasser, Abfall, Stadtwerke, Zählerstand)
     - `Telekommunikation` (Mobilfunk, Internet, DSL, Telefon, Festnetz, Router)
     - `Gesundheit` (Arzt, Krankenhaus, Apotheke, Medikamente, Befund, Krankenkasse)
     - `Versicherung` (Haftpflicht, Hausrat, Rechtsschutz, Lebensversicherung, Schadensmeldung)
     - `Immobilien` (Mietvertrag, Nebenkosten, Hausgeld, Grundsteuer, Hausverwaltung, WEG)
     - `Gehalt` (Gehaltsabrechnung, Lohnsteuerbescheinigung, Sozialversicherung)
     - `Bank` (Kontoauszug, Darlehen, Kredit, Zinsen, Gebühren, Kreditkarte)
     - `Depot` (Wertpapiere, Aktien, Fonds, Dividende, Kaufabrechnung, WKN, ISIN)
     - `Behörde` (Finanzamt, Stadtverwaltung, Bürgeramt, Rentenversicherung, Amtsgericht)
     - `Bildung` (Schule, Universität, Weiterbildung, Kurse, Zertifikat)
     - `Reisen` (Hotel, Flug, Bahn, Buchung, Ferienwohnung, Fahrkarte)
     - `Mitgliedschaft` (Verein, Verband, Gewerkschaft, ADAC, Clubbeitrag)
     - `Altersvorsorge` (Rente, Versorgungswerk, Riester, Rürup, Betriebsrente)

## Acceptance Criteria

- [ ] Alle 12 kanonischen Dokumenttypen verfügen über mehrzeilige, präzise und layout-spezifische deutsche Beschreibungen.
- [ ] Alle 5 steuerlichen Abzugsklassen sind mit differenzierten Begriffen und rechtlichen Kontexten abgebildet.
- [ ] Alle 14 thematischen Tag-Taxonomie-Klassen verfügen über detaillierte Kontextanker und Alttag-Cluster.
- [ ] Helper-Funktionen `get_document_type_anchors()`, `get_tax_anchors()` und `get_thematic_tag_anchors()` stehen zur Verfügung.
- [ ] Pydantic-/Dataclass-Validierung stellt sicher, dass keine leeren Beschreibungen existieren.
- [ ] Unit-Tests verifizieren die Vollständigkeit des Katalogs.
