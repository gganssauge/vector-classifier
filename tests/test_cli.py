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


def test_cli_commands_help():
    """Verify CLI subcommands --help display properly."""
    for cmd in ["calibrate-tax", "tag-candidates", "sync-curated"]:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0
        assert "Usage: vector-classifier" in result.stdout


def test_cli_calibrate_tax_command(monkeypatch):
    """Verify calibrate-tax command invokes TaxCalibrationRunner."""
    from unittest.mock import AsyncMock, MagicMock

    mock_runner = MagicMock()
    mock_runner.run = AsyncMock(
        return_value={
            "curated_2020_2025": 10,
            "nontax_2020_2025": 20,
            "historical_pre_2020_tax": 5,
            "historical_pre_2020_nontax": 15,
            "post_2025_total": 30,
            "post_2025_tax_candidates": 2,
        }
    )
    mock_runner.close = AsyncMock()

    monkeypatch.setattr(
        "vector_classifier.calibrate.TaxCalibrationRunner",
        lambda **kwargs: mock_runner,
    )

    result = runner.invoke(app, ["calibrate-tax"])
    assert result.exit_code == 0
    assert "Summary of 3-Zone Calibration:" in result.stdout


def test_cli_tag_candidates_command(monkeypatch):
    """Verify tag-candidates command invokes TaxReviewManager.tag_candidates."""
    from unittest.mock import AsyncMock, MagicMock

    mock_mgr = MagicMock()
    mock_mgr.tag_candidates = AsyncMock(
        return_value={
            "candidates": 5,
            "updated": 5,
            "already_tagged": 0,
            "errors": 0,
        }
    )
    mock_mgr.close = AsyncMock()

    monkeypatch.setattr(
        "vector_classifier.tax_review.TaxReviewManager",
        lambda **kwargs: mock_mgr,
    )

    result = runner.invoke(app, ["tag-candidates", "--dry-run"])
    assert result.exit_code == 0
    assert "Candidate Tagging Complete:" in result.stdout


def test_cli_sync_curated_command(monkeypatch):
    """Verify sync-curated command invokes TaxReviewManager.sync_curated."""
    from unittest.mock import AsyncMock, MagicMock

    mock_mgr = MagicMock()
    mock_mgr.sync_curated = AsyncMock(
        return_value={
            "total_candidates": 5,
            "confirmed": 4,
            "rejected": 1,
            "pending": 0,
            "errors": 0,
        }
    )
    mock_mgr.close = AsyncMock()

    monkeypatch.setattr(
        "vector_classifier.tax_review.TaxReviewManager",
        lambda **kwargs: mock_mgr,
    )

    result = runner.invoke(app, ["sync-curated", "--dry-run"])
    assert result.exit_code == 0
    assert "Curated Synchronization Complete:" in result.stdout
