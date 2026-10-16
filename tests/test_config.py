"""Tests for application configuration."""

from vector_classifier.config import Settings


def test_default_settings():
    """Verify default settings values."""
    s = Settings()
    assert "ollama" in s.ollama_base_url
    assert s.ollama_embed_model == "bge-m3"
    assert s.threshold_tax_confident == 0.82
    assert s.threshold_tax_review == 0.70
    assert s.threshold_tag == 0.78
    assert s.threshold_doctype_confident == 0.80
