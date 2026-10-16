"""Tests for semantic anchor catalogs."""

from vector_classifier.vectors.anchors import (
    get_document_type_anchors,
    get_tax_anchors,
    get_thematic_tag_anchors,
)


def test_document_type_anchors_completeness():
    """Verify document type anchors are populated and valid."""
    anchors = get_document_type_anchors()
    assert len(anchors) >= 11
    names = [a.name for a in anchors]
    assert "Rechnung" in names
    assert "Abrechnung" in names
    assert "Kontoauszug" in names
    assert "Bescheid" in names
    for a in anchors:
        assert len(a.description.strip()) > 30
        assert len(a.keywords) > 0


def test_tax_anchors_completeness():
    """Verify tax anchors cover all 5 required categories."""
    anchors = get_tax_anchors()
    assert len(anchors) == 5
    names = [a.name for a in anchors]
    assert "steuer:werbungskosten" in names
    assert "steuer:haushaltsnah-35a" in names
    assert "steuer:sonderausgaben" in names
    assert "steuer:krankheit-aussergewoehnlich" in names
    assert "steuer:anlage-v" in names
    for a in anchors:
        assert len(a.description.strip()) > 30


def test_thematic_tag_anchors_completeness():
    """Verify 14 thematic macro-tag anchors."""
    anchors = get_thematic_tag_anchors()
    assert len(anchors) == 14
    names = [a.name for a in anchors]
    assert "Fahrzeug" in names
    assert "Energie" in names
    assert "Telekommunikation" in names
    assert "Immobilien" in names
    for a in anchors:
        assert len(a.description.strip()) > 10
