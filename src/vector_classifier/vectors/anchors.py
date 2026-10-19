"""Semantic anchor catalog for document types, tax categories, and thematic tags."""

from vector_classifier.models.schemas import SemanticAnchor

DOCUMENT_TYPE_ANCHORS: list[SemanticAnchor] = [
    SemanticAnchor(
        name="Rechnung",
        category="doc_type",
        description=(
            "Kaufbelege, Handelsrechnungen und Versorgerrechnungen über gelieferte Waren oder erbrachte Leistungen "
            "mit einer Zahlungsaufforderung oder Geldforderung. Dazu gehören Rechnungen und Schlussrechnungen von Energieversorgern "
            "(Strom, Gas, Wasser, Fernwärme), Telekommunikations- und Telefonrechnungen (Mobilfunk, Internet), "
            "Handwerker- und Reparaturrechnungen, Arztrechnungen, Therapeuten- und Apothekenrechnungen sowie "
            "Depotführungs- und Dienstleistungsrechnungen. Enthält typischerweise Rechnungsnummer, Leistungszeitraum, "
            "Rechnungsbetrag, Zahlungsziel, Fälligkeit oder Bankverbindung."
        ),
        keywords=[
            "rechnung",
            "invoice",
            "rechnungsnummer",
            "rechnungsbetrag",
            "schlussrechnung",
            "stromrechnung",
            "gasrechnung",
            "telefonrechnung",
            "handwerkerrechnung",
            "arztrechnung",
            "zahlungsziel",
            "faelligkeit",
            "gesamtbetrag",
            "umsatzsteuer",
            "kaufbeleg",
        ],
    ),
    SemanticAnchor(
        name="Abrechnung",
        category="doc_type",
        description=(
            "Periodische Verrechnungs- und Abrechnungsübersichten ohne unmittelbaren Handelsrechnungscharakter: "
            "monatliche Lohn- und Gehaltsabrechnungen, Entgeltnachweise der Arbeitgeber, jährliche Hausgeld- und "
            "Nebenkostenabrechnungen der Hausverwaltung oder des Vermieters (mit Verteilerschlüssel, Vorauszahlungsabgleich "
            "und Umlageberechnung), Reisekostenabrechnungen, Kreditkartenabrechnungen (Mastercard, Visa) sowie "
            "Bankabrechnungen über Wertpapiere: Wertpapierabrechnung, Wertpapierverkauf, Wertpapierkauf, steuerliche Behandlung von Wertpapierkäufen und Wertpapierverkäufen, Orderabrechnung, "
            "Depotübersicht, Ertragsabrechnung, Ertragsgutschrift, Dividendengutschrift, Zinsabrechnung oder Steuermitteilung der Bank. "
            "Keine reinen Girokontoauszüge und keine Versorgerrechnungen."
        ),
        keywords=[
            "abrechnung",
            "gehalt",
            "lohn",
            "entgeltnachweis",
            "bezuege",
            "hausgeldabrechnung",
            "nebenkostenabrechnung",
            "betriebskostenabrechnung",
            "reisekostenabrechnung",
            "kreditkartenabrechnung",
            "wertpapierabrechnung",
            "wertpapierverkauf",
            "wertpapierkauf",
            "steuerliche behandlung",
            "kapitalertragsteuer",
            "solidaritaetszuschlag",
            "steuerbemessungsgrundlage",
            "ertragsabrechnung",
            "ertragsgutschrift",
            "dividendengutschrift",
            "zinsabrechnung",
            "orderabrechnung",
            "depotabrechnung",
            "steuermitteilung",
        ],
    ),
    SemanticAnchor(
        name="Kontoauszug",
        category="doc_type",
        description=(
            "Fortlaufende periodische Finanzübersichten und Saldennachweise eines Girokontos, Tagesgeldkontos "
            "sowie eines Hausabrechnungs- oder WEG-Abrechnungskontos einer Eigentümergemeinschaft mit Auszugsnummer, "
            "Buchungstag, Wertstellung, Buchungstext, Kontostand, Umsätzen, Anfangssaldo und Schlusssaldo sowie IBAN. "
            "Keine Wertpapierabrechnungen, keine steuerlichen Abrechnungen von Wertpapierkäufen und keine reinen Bankmitteilungen."
        ),
        keywords=[
            "kontoauszug",
            "girokonto",
            "abrechnungskonto",
            "weg-konto",
            "auszugsnummer",
            "kontostand",
            "buchungstag",
            "wertstellung",
            "saldo",
            "anfangssaldo",
            "schlusssaldo",
            "umsatzanzeige",
            "umsatzuebersicht",
        ],
    ),
    SemanticAnchor(
        name="Bankmitteilung",
        category="doc_type",
        description=(
            "Mitteilungen, Anschreiben, Informationsschreiben und Auskünfte von Banken, Sparkassen oder Brokern. "
            "Beispielsweise Mitteilungen über Freistellungsaufträge, Nichtveranlagungsbescheinigungen, Zins- und Konditionsanpassungen, "
            "AGB-Änderungen, Depot-Performanceberichte oder allgemeine Kundenkorrespondenz eines Kreditinstituts. "
            "Keine periodischen Kontoauszüge mit Salden und keine Wertpapierabrechnungen."
        ),
        keywords=[
            "bankmitteilung",
            "freistellungsauftrag",
            "nichtveranlagungsbescheinigung",
            "zinsanpassung",
            "konditionenaenderung",
            "bankauskunft",
            "depotmitteilung",
            "performancebericht",
            "commerzbank",
            "deutsche bank",
            "targobank",
            "apobank",
        ],
    ),
    SemanticAnchor(
        name="Mahnung",
        category="doc_type",
        description=(
            "Zahlungserinnerungen, Mahnungen, letzte Mahnschreiben, Inkassobenachrichtigungen und Verzugsschreiben "
            "wegen fälliger Geldforderungen sowie behördliche oder vertragliche Erinnerungen und Mahnungen "
            "zur Einreichung von Unterlagen oder Erteilung von Auskünften."
        ),
        keywords=[
            "mahnung",
            "zahlungserinnerung",
            "inkasso",
            "letzte mahnung",
            "verzug",
            "mahngebuehr",
            "zahlungsausstand",
            "offener betrag",
            "mahnstufe",
            "verzugszinsen",
        ],
    ),
    SemanticAnchor(
        name="Dokumentation",
        category="doc_type",
        description=(
            "Technische Dokumentationen, Bedienungsanleitungen, Handbücher, Benutzerhandbücher, "
            "Produkt- und Sicherheitsdatenblätter, Kalibrierungsberichte, Prüfberichte, Messprotokolle oder Geräteunterlagen. "
            "Keine geschäftliche Korrespondenz und keine Verträge."
        ),
        keywords=[
            "dokumentation",
            "handbuch",
            "bedienungsanleitung",
            "benutzerhandbuch",
            "datenblatt",
            "datenblaetter",
            "kalibrierungsbericht",
            "pruefbericht",
            "messprotokoll",
            "anleitung",
            "geraeteunterlagen",
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
            "Offizielle Bestätigungen, Nachweise und Bescheinigungen ohne unmittelbare neue Zahlungsaufforderung. "
            "Dazu gehören Bank-Überweisungsbestätigungen, Ausführungsbestätigungen, Zahlungsnachweise, "
            "Zuwendungsbestätigungen, Spendenbescheinigungen, Meldebescheinigungen, Immatrikulationsbescheinigungen, "
            "Arbeits- und Versicherungsbestätigungen. Kein periodischer Kontoauszug mit Salden."
        ),
        keywords=[
            "bescheinigung",
            "bestaetigung",
            "nachweis",
            "ueberweisungsbestaetigung",
            "ausfuehrungsbestaetigung",
            "zahlungsnachweis",
            "zuwendungsbestaetigung",
            "spendenbescheinigung",
        ],
    ),
    SemanticAnchor(
        name="Vertrag",
        category="doc_type",
        description=(
            "Rechtsgeschäftliche Vereinbarungen und Vertragswerke zwischen zwei oder mehr Parteien. "
            "Beispielsweise Mietvertrag (auch mit Nennung von Kaltmiete, Kaution und Betriebskostenvorauszahlung), "
            "Arbeitsvertrag, Kaufvertrag, Darlehensvertrag, Versicherungsschein, Police, Nachträge und Vertragsänderungen. "
            "Keine periodische Abrechnung."
        ),
        keywords=[
            "vertrag",
            "vereinbarung",
            "mietvertrag",
            "arbeitsvertrag",
            "police",
            "versicherungsschein",
            "vertragsbedingungen",
            "vertragsdauer",
            "kuendigung",
            "unterschrift",
        ],
    ),
    SemanticAnchor(
        name="Korrespondenz",
        category="doc_type",
        description=(
            "Allgemeiner schriftlicher Briefverkehr, formlose Mitteilungen, Anschreiben, Informationsschreiben "
            "oder Kundenbetreuungsschreiben ohne unmittelbaren Rechnungs-, Bescheid- oder Vertragscharakter. "
            "Umfasst auch behördliche Korrespondenz, Schreiben und Auskunftsersuchen des Finanzamts, der Krankenkasse "
            "oder des Steuerberaters, sofern es sich nicht um einen förmlichen Bescheid mit Rechtsbehelfsbelehrung handelt."
        ),
        keywords=[
            "anschreiben",
            "mitteilung",
            "sehr geehrte",
            "mit freundlichen gruessen",
            "brief",
            "nachricht",
            "finanzamt",
            "steuerberater",
            "auskunftsersuchen",
            "betreff",
        ],
    ),
    SemanticAnchor(
        name="Steuern",
        category="doc_type",
        description=(
            "Förmliche Steuererklärungsunterlagen, Elster-Übertragungsprotokolle, Mantelbögen, Anlagen zur Steuererklärung, "
            "amtliche Steuerberechnungsübersichten und Fragebögen zur steuerlichen Erfassung. "
            "Reine Behördenschreiben und Briefe gehören zur Korrespondenz, förmliche Festsetzungen zum Bescheid."
        ),
        keywords=[
            "steuererklaerung",
            "elster",
            "uebertragungsprotokoll",
            "mantelbogen",
            "steuernummer",
            "anlage eür",
            "anlage v",
        ],
    ),
    SemanticAnchor(
        name="Gesundheit",
        category="doc_type",
        description=(
            "Medizinische Unterlagen, ärztliche Befunde, Laborberichte, Arztbriefe, Entlassungsberichte, "
            "Rezepte, Verordnungen, Impfausweise oder Patientenverfügungen. "
            "Keine Rechnungen von Ärzten, Therapeuten oder Apotheken (diese sind Dokumenttyp Rechnung mit thematischem Tag Gesundheit)."
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
            "Sonstige Unterlagen, Dokumentationen, Notizen oder Informationsmaterialien, "
            "die in keine der regulären Primärkategorien eingeordnet werden können."
        ),
        keywords=[
            "sonstiges",
            "notiz",
            "information",
        ],
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
        description="Strom, Gas, Fernwärme, Wasser, Abwasser, Stadtwerke, Berliner Wasserbetriebe, GASAG, Energieversorger, Zählerstand, Zählernummer, kWh, Kubikmeter m³, Jahresabrechnung Energie, Wasserrechnung.",
        keywords=[
            "strom",
            "gas",
            "fernwaerme",
            "stadtwerke",
            "zaehlerstand",
            "wasser",
            "abwasser",
            "kwh",
            "energie",
        ],
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
        description="Mietvertrag, Mietwohnung, Nebenkosten, Hausverwaltung, Hausgeld, Grundsteuer, Eigentümergemeinschaft WEG, Schornsteinfeger, Schornsteinfegermeisterbetrieb, Feuerstättenbescheid, Kamin, Handwerker, Hausinstandhaltung, Renovierung, Dach, Sanitär, Maler, Gebäudeunterhalt, Grundbuch, Eigentumswohnung, Wohnhaus, Hauswart.",
        keywords=[
            "wohnung",
            "miete",
            "hausverwaltung",
            "hausgeld",
            "grundsteuer",
            "schornsteinfeger",
            "feuerstaettenbescheid",
            "handwerker",
            "instandhaltung",
            "grundbuch",
        ],
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
        description="Kreditinstitute, Bankkonten, Girokonto, Kontokorrent, Deutsche Apotheker- und Ärztebank (Apobank), Commerzbank, Deutsche Bank, Sparkasse, ING, DKB, Postbank, Darlehen, Kredit, Ratenkredit, Baufinanzierung, Zinsen, Sollzins, Bankgebühren, Kreditkarte, Überweisung, Dauerauftrag, Auszugsnummer, Kontostand, Saldo, IBAN.",
        keywords=[
            "bank",
            "girokonto",
            "kontokorrent",
            "kredit",
            "darlehen",
            "zinsen",
            "gebuehren",
            "apobank",
            "aerztebank",
            "ueberweisung",
            "iban",
            "saldo",
        ],
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
        description="Schule, Hochschule, Universität, Studium, Weiterbildung, Schulung, Zertifikat, Diplom, Lehrgang, Fachliteratur, Fachverlag, Fachbuch, Fachzeitschrift, Heise Medien, Haufe Group, Computerzeitschrift, c't, iX, Fachinformationen, Abonnement.",
        keywords=[
            "schule",
            "universitaet",
            "studium",
            "weiterbildung",
            "zertifikat",
            "fachliteratur",
            "fachverlag",
            "heise",
            "haufe",
            "fachzeitschrift",
            "abonnement",
        ],
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
