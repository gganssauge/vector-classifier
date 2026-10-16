"""Tests for Typer CLI."""

from typer.testing import CliRunner

from vector_classifier import __version__
from vector_classifier.cli import app

runner = CliRunner()


def test_cli_version():
    """Test CLI version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_info():
    """Test CLI info command."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Ollama URL" in result.stdout
    assert "Qdrant Host" in result.stdout
