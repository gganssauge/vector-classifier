"""Semantic anchor catalog for document types, tax categories, and thematic tags."""

from vector_classifier.models.schemas import SemanticAnchor

DOCUMENT_TYPE_ANCHORS: list[SemanticAnchor] = [
    SemanticAnchor(
        name="Rechnung",
        category="doc_type",
        description=(
            "Kaufbelege und Handelsrechnungen über gelieferte Waren oder erbrachte Dienstleistungen. "
            "Enthält typischerweise Rechnungsnummer, Leistungszeitraum, Zahlungsziel, Umsatzsteuer, "
            "Mehrwertsteuer, Einzelpositionen, Gesamtbetrag, Zahlungsanweisung oder IBAN."
        ),
        keywords=[
            "rechnung",
            "invoice",
            "gesamtbetrag",
            "rechnungsnummer",
            "zahlungsziel",
            "umsatzsteuer",
        ],
    ),
    SemanticAnchor(
        name="Abrechnung",
        category="doc_type",
        description=(
            "Wiederkehrende Abrechnungen und Periodenabrechnungen wie Nebenkosten-, Betriebs- und "
            "Heizkostenabrechnung, Hausgeldabrechnung, monatliche Lohn- oder Gehaltsabrechnung, "
            "Entgeltnachweis sowie Kreditkarten- oder Zinsabrechnungen."
        ),
        keywords=[
            "abrechnung",
            "nebenkosten",
            "betriebskosten",
            "hausgeld",
            "gehalt",
            "lohn",
            "entgelt",
        ],
    ),
    SemanticAnchor(
        name="Kontoauszug",
        category="doc_type",
        description=(
            "Bankauszüge und Finanzübersichten eines Giro-, Tagesgeld- oder Anlagekontos. "
            "Enthält Buchungstag, Wertstellung, Buchungstext, Kontostand, Umsätze, "
            "Anfangs- und Schlusssaldo sowie IBAN und BIC."
        ),
        keywords=[
            "kontoauszug",
            "kontostand",
            "buchungstag",
            "wertstellung",
            "saldo",
            "umsatzanzeige",
        ],
    ),
    SemanticAnchor(
        name="Bescheid",
        category="doc_type",
        description=(
            "Behördliche Festsetzungen und öffentlich-rechtliche Verwaltungsakte mit Rechtsbehelfsbelehrung. "
            "Beispielsweise Einkommensteuerbescheid, Grundsteuerbescheid, Gebührenbescheid, "
            "Bußgeldbescheid, Rentenbescheid oder Beitragsbescheid."
        ),
        keywords=[
            "bescheid",
            "festsetzung",
            "rechtsbehelfsbelehrung",
            "verwaltungsakt",
            "finanzamt",
            "festgesetzt",
        ],
    ),
    SemanticAnchor(
        name="Bescheinigung",
        category="doc_type",
        description=(
            "Offizielle Bestätigungen und Nachweise ohne unmittelbare Zahlungsaufforderung. "
            "Dazu gehören Spendenbescheinigungen, Zuwendungsbestätigungen, Immatrikulationsbescheinigungen, "
            "Arbeitsbescheinigungen, Meldebescheinigungen oder Versicherungsbestätigungen."
        ),
        keywords=[
            "bescheinigung",
            "bestaetigung",
            "nachweis",
            "zuwendungsbestaetigung",
            "spendenbescheinigung",
        ],
    ),
    SemanticAnchor(
        name="Vertrag",
        category="doc_type",
        description=(
            "Rechtsgeschäftliche Vereinbarungen zwischen zwei oder mehr Parteien. "
            "Beispielsweise Mietvertrag, Arbeitsvertrag, Kaufvertrag, Versicherungsschein, Police, "
            "allgemeine Geschäftsbedingungen sowie rechtsverbindliche Kündigungen oder Nachträge."
        ),
        keywords=[
            "vertrag",
            "vereinbarung",
            "police",
            "versicherungsschein",
            "vertragsbedingungen",
            "unterschrift",
        ],
    ),
    SemanticAnchor(
        name="Korrespondenz",
        category="doc_type",
        description=(
            "Allgemeiner schriftlicher Briefverkehr, formlose Mitteilungen, Anschreiben, Informationsschreiben "
            "oder Kundenbetreuungsschreiben ohne unmittelbaren Rechnungs-, Bescheid- oder Vertragscharakter."
        ),
        keywords=[
            "anschreiben",
            "mitteilung",
            "sehr geehrte",
            "mit freundlichen gruessen",
            "brief",
            "nachricht",
        ],
    ),
    SemanticAnchor(
        name="Steuern",
        category="doc_type",
        description=(
            "Dokumente der Steuererklärung und Steuerkorrespondenz, Elster-Übertragungsprotokolle, "
            "Mitteilungen über Steuervorauszahlungen, steuerliche Fragebögen und Anträge an das Finanzamt."
        ),
        keywords=[
            "steuererklaerung",
            "elster",
            "finanzamt",
            "steuernummer",
            "vorauszahlung",
            "steuerberechnung",
        ],
    ),
    SemanticAnchor(
        name="Gesundheit",
        category="doc_type",
        description=(
            "Medizinische Unterlagen, ärztliche Befunde, Laborberichte, Arztbriefe, Entlassungsberichte, "
            "Rezepte, Verordnungen, Impfausweise oder Patientenverfügungen."
        ),
        keywords=[
            "befund",
            "diagnose",
            "arztbrief",
            "laborwerte",
            "anamnese",
            "patient",
            "klinik",
        ],
    ),
    SemanticAnchor(
        name="Urkunde",
        category="doc_type",
        description=(
            "Notarielle und amtliche Beurkundungen wie Grundbuchauszüge, Geburtsurkunden, Heiratsurkunden, "
            "Sterbeurkunden, Gesellschafterbeschlüsse oder notarielle Vollmachten."
        ),
        keywords=[
            "urkunde",
            "notar",
            "beurkundung",
            "grundbuch",
            "standesamt",
            "beglaubigung",
        ],
    ),
    SemanticAnchor(
        name="Sonstiges",
        category="doc_type",
        description=(
            "Sonstige Unterlagen, Dokumentationen, Anleitungen, Notizen oder Werbematerialien, "
            "die in keine der regulären Primärkategorien eingeordnet werden können."
        ),
        keywords=["sonstiges", "anleitung", "notiz", "information", "dokumentation"],
    ),
]

TAX_ANCHORS: list[SemanticAnchor] = [
    SemanticAnchor(
        name="steuer:werbungskosten",
        category="tax",
        description=(
            "Aufwendungen und Ausgaben zur Erwerbung, Sicherung und Erhaltung von Einnahmen aus nichtselbstständiger "
            "Arbeit. Belege über Arbeitsmittel wie Computer, Monitore, Notebooks, Tastaturen, Software-Lizenzen, "
            "Bürobedarf, Fachliteratur, Fachbücher, berufliche Fortbildungen, Kurse und Fahrtkosten."
        ),
        keywords=[
            "arbeitsmittel",
            "fachliteratur",
            "fortbildung",
            "bueroausstattung",
            "hardware",
            "werbungskosten",
        ],
    ),
    SemanticAnchor(
        name="steuer:haushaltsnah-35a",
        category="tax",
        description=(
            "Handwerkerleistungen und haushaltsnahe Dienstleistungen nach § 35a EStG für Renovierung, Erhaltung, "
            "Modernisierung oder Pflege im Privathaushalt. Rechnungen von Handwerkern mit getrennt ausgewiesenen "
            "Lohnkosten, Arbeitskosten, Schornsteinfeger, Maler, Sanitär, Gartenpflege und Gebäudereinigung."
        ),
        keywords=[
            "35a",
            "handwerkerleistung",
            "arbeitskosten",
            "lohnanteil",
            "schornsteinfeger",
            "haushaltsnah",
        ],
    ),
    SemanticAnchor(
        name="steuer:sonderausgaben",
        category="tax",
        description=(
            "Vorsorgeaufwendungen und gesetzlich begünstigte Privatausgaben wie Beiträge zur Kranken- und Pflegeversicherung, "
            "Rentenversicherung, Rürup, Riester, Haftpflichtversicherung sowie Spenden und Zuwendungen an gemeinnützige "
            "Organisationen und Kirchensteuer."
        ),
        keywords=[
            "sonderausgaben",
            "spende",
            "zuwendungsbestaetigung",
            "vorsorgeaufwand",
            "kirchensteuer",
            "riester",
        ],
    ),
    SemanticAnchor(
        name="steuer:krankheit-aussergewoehnlich",
        category="tax",
        description=(
            "Außergewöhnliche Belastungen aus Krankheits- und Heilbehandlungskosten. Rechnungen für Arztbesuche, Zahnbehandlungen, "
            "Zahnersatz, verschreibungspflichtige Medikamente aus der Apotheke, Sehhilfen, Brillen, Hörgeräte, "
            "physiotherapeutische Behandlungen und Zuzahlungen."
        ),
        keywords=[
            "krankheitskosten",
            "arztrechnung",
            "apotheke",
            "zahnersatz",
            "heilbehandlung",
            "aussergewoehnliche belastung",
        ],
    ),
    SemanticAnchor(
        name="steuer:anlage-v",
        category="tax",
        description=(
            "Einkünfte aus Vermietung und Verpachtung sowie Belege für vermietete Immobilien. "
            "Hausgeldabrechnungen, Grundsteuer, Gebäudeversicherungen, Erhaltungsaufwand, Instandhaltung "
            "und Handwerkerrechnungen für Mietsachobjekte."
        ),
        keywords=[
            "anlage v",
            "vermietung",
            "hausgeldabrechnung",
            "gebaeudeversicherung",
            "grundsteuer",
            "instandhaltung",
        ],
    ),
]

THEMATIC_TAG_ANCHORS: list[SemanticAnchor] = [
    SemanticAnchor(
        name="Fahrzeug",
        category="theme",
        description="Kraftfahrzeug, Pkw, Auto, Kfz-Werkstatt, Reparatur, Tanken, TÜV, Hauptuntersuchung, Leasing, Reifen, Kfz-Zulassung.",
        keywords=["kfz", "auto", "werkstatt", "tuev", "tanken", "leasing"],
    ),
    SemanticAnchor(
        name="Energie",
        category="theme",
        description="Strom, Gas, Fernwärme, Wasser, Abwasser, Stadtwerke, Energieversorger, Zählerstand, Jahresabrechnung Energie.",
        keywords=["strom", "gas", "fernwaerme", "stadtwerke", "zaehlerstand"],
    ),
    SemanticAnchor(
        name="Telekommunikation",
        category="theme",
        description="Mobilfunk, Internet, DSL, Breitband, Festnetz, Telefonanbieter, Router, Glasfaser, Datenvolumen.",
        keywords=["mobilfunk", "internet", "dsl", "glasfaser", "telekom", "vodafone"],
    ),
    SemanticAnchor(
        name="Gesundheit",
        category="theme",
        description="Medizin, Arzt, Facharzt, Zahnarzt, Krankenhaus, Befunde, Apotheke, Medikamente, Krankenkasse.",
        keywords=["arzt", "krankenhaus", "apotheke", "medikament", "krankenkasse"],
    ),
    SemanticAnchor(
        name="Versicherung",
        category="theme",
        description="Versicherungsverträge, Haftpflicht, Hausrat, Rechtsschutz, Kfz-Versicherung, Lebensversicherung, Schadensanzeige.",
        keywords=[
            "versicherung",
            "police",
            "haftpflicht",
            "hausrat",
            "schadensmeldung",
        ],
    ),
    SemanticAnchor(
        name="Immobilien",
        category="theme",
        description="Mietvertrag, Mietwohnung, Nebenkosten, Hausverwaltung, Hausgeld, Grundsteuer, Eigentümergemeinschaft WEG.",
        keywords=["wohnung", "miete", "hausverwaltung", "hausgeld", "grundsteuer"],
    ),
    SemanticAnchor(
        name="Gehalt",
        category="theme",
        description="Arbeitgeber, Lohn- und Gehaltsabrechnung, Entgeltabrechnung, Lohnsteuerbescheinigung, Sozialversicherungsmeldung.",
        keywords=["gehalt", "lohn", "arbeitgeber", "entgelt", "sozialversicherung"],
    ),
    SemanticAnchor(
        name="Bank",
        category="theme",
        description="Bankkonten, Girokonto, Darlehen, Kredit, Ratenkredit, Baufinanzierung, Zinsen, Bankgebühren, Kreditkarte.",
        keywords=["bank", "girokonto", "kredit", "darlehen", "zinsen", "gebuehren"],
    ),
    SemanticAnchor(
        name="Depot",
        category="theme",
        description="Wertpapierdepot, Aktien, Fonds, ETF, Wertpapierabrechnung, Kauf, Verkauf, Dividende, ISIN, WKN.",
        keywords=["depot", "aktien", "fonds", "etf", "dividende", "isin", "wertpapier"],
    ),
    SemanticAnchor(
        name="Behörde",
        category="theme",
        description="Öffentliche Verwaltung, Bürgeramt, Stadtverwaltung, Finanzamt, Deutsche Rentenversicherung, Amtsgericht, Meldeamt.",
        keywords=[
            "behoerde",
            "buergeramt",
            "stadtverwaltung",
            "amtsgericht",
            "rentenversicherung",
        ],
    ),
    SemanticAnchor(
        name="Bildung",
        category="theme",
        description="Schule, Hochschule, Universität, Studium, Weiterbildung, Schulung, Zertifikat, Diplom, Lehrgang.",
        keywords=["schule", "universitaet", "studium", "weiterbildung", "zertifikat"],
    ),
    SemanticAnchor(
        name="Reisen",
        category="theme",
        description="Urlaub, Reisen, Hotel, Unterkunft, Flug, Deutsche Bahn, Zugticket, Buchungsbestätigung, Ferienwohnung.",
        keywords=["reise", "hotel", "flug", "bahn", "urlaub", "buchung"],
    ),
    SemanticAnchor(
        name="Mitgliedschaft",
        category="theme",
        description="Vereine, Verbände, Gewerkschaften, Automobilclub ADAC, Clubbeiträge, Mitgliedsbeitrag.",
        keywords=[
            "verein",
            "verband",
            "mitgliedschaft",
            "adac",
            "gewerkschaft",
            "beitrag",
        ],
    ),
    SemanticAnchor(
        name="Altersvorsorge",
        category="theme",
        description="Gesetzliche Rente, Renteninformation, Versorgungswerk, Betriebsrente, bAV, private Altersvorsorge, Rürup, Riester.",
        keywords=[
            "rente",
            "altersvorsorge",
            "renteninformation",
            "versorgungswerk",
            "betriebsrente",
        ],
    ),
]


def get_document_type_anchors() -> list[SemanticAnchor]:
    """Return canonical semantic anchors for document types."""
    return DOCUMENT_TYPE_ANCHORS


def get_tax_anchors() -> list[SemanticAnchor]:
    """Return semantic anchors for tax categories."""
    return TAX_ANCHORS


def get_thematic_tag_anchors() -> list[SemanticAnchor]:
    """Return semantic anchors for thematic tags."""
    return THEMATIC_TAG_ANCHORS
