"""Tests for application configuration."""

from vector_classifier.config import Settings


def test_default_settings():
    """Verify default settings values."""
    s = Settings(_env_file=None)
    assert "ollama" in s.ollama_base_url
    assert s.ollama_embed_model == "bge-m3"
    assert s.threshold_tax_confident == 0.60
    assert s.threshold_tax_review == 0.52
    assert s.threshold_tag == 0.50
    assert s.threshold_doctype_confident == 0.50
