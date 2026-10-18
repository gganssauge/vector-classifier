"""Command-line interface for vector-classifier."""

import typer
from rich.console import Console

from vector_classifier import __version__
from vector_classifier.config import settings

app = typer.Typer(
    name="vector-classifier",
    help="Semantic Vector Classification and Search CLI",
    add_completion=False,
)
console = Console()
DRY_RUN_OPTION: str = "--dry-run/--no-dry-run"


@app.command()
def version() -> None:
    """Print the service version."""
    console.print(f"vector-classifier version [bold green]{__version__}[/bold green]")


@app.command()
def info() -> None:
    """Display current service configuration."""
    console.print("[bold cyan]Vector Classifier Configuration:[/bold cyan]")
    console.print(f"  Ollama URL:        {settings.ollama_base_url}")
    console.print(f"  Ollama Model:      {settings.ollama_embed_model}")
    console.print(f"  Qdrant Host:       {settings.qdrant_host}:{settings.qdrant_port}")
    console.print(f"  Qdrant Collection: {settings.qdrant_collection}")
    console.print(f"  Tax Threshold:     {settings.threshold_tax_confident}")
    console.print(f"  Tag Threshold:     {settings.threshold_tag}")


@app.command()
def run(
    host: str = typer.Option(settings.host, "--host", "-h", help="Bind host"),
    port: int = typer.Option(settings.port, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable live reload"),
) -> None:
    """Run the FastAPI web server."""
    import uvicorn

    console.print(f"[bold green]Starting server on {host}:{port}...[/bold green]")
    uvicorn.run("vector_classifier.main:app", host=host, port=port, reload=reload)


@app.command()
def audit(
    source: str = typer.Option(
        ...,
        "--source",
        "-s",
        help="Path to input JSON or JSONL document file",
    ),
    output_csv: str = typer.Option(
        "audit_report.csv",
        "--output-csv",
        "-o",
        help="Path for tabular CSV evaluation output",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit number of documents to process",
    ),
    dry_run: bool = typer.Option(
        True,
        DRY_RUN_OPTION,
        help="Dry-run mode: do not write changes to Paperless",
    ),
    concurrency: int = typer.Option(
        4,
        "--concurrency",
        "-c",
        help="Maximum concurrent embedding requests",
    ),
    index: bool = typer.Option(
        False,
        "--index/--no-index",
        help="Upsert vectorized documents into Qdrant",
    ),
) -> None:
    """Run batch classification and offline audit evaluation on documents."""
    import asyncio

    from vector_classifier.audit import AuditRunner

    console.print("[bold cyan]Starting Vector Classifier Audit Pass...[/bold cyan]")
    console.print(f"  Source file:  {source}")
    console.print(f"  Output CSV:   {output_csv}")
    console.print(f"  Dry run:      {dry_run}")
    console.print(f"  Index Qdrant: {index}")
    console.print(f"  Concurrency:  {concurrency}")
    if limit is not None:
        console.print(f"  Limit:        {limit}")

    runner = AuditRunner(
        concurrency=concurrency,
        index=index,
        dry_run=dry_run,
    )

    async def _execute() -> None:
        try:
            await runner.run(
                source=source,
                output_csv=output_csv,
                limit=limit,
                console=console,
            )
        finally:
            await runner.close()

    asyncio.run(_execute())


@app.command()
def ingest(
    url: str = typer.Option(
        settings.paperless_base_url,
        "--url",
        "-u",
        help="Paperless REST API Base URL",
    ),
    token: str = typer.Option(
        settings.paperless_api_token,
        "--token",
        "-t",
        help="Paperless API Authorization Token",
    ),
    concurrency: int = typer.Option(
        8,
        "--concurrency",
        "-c",
        help="Concurrent embedding requests to Ollama",
    ),
    batch_size: int = typer.Option(
        50,
        "--batch-size",
        "-b",
        help="Number of documents per Qdrant upsert batch",
    ),
    recreate: bool = typer.Option(
        False,
        "--recreate/--no-recreate",
        help="Recreate Qdrant collection fresh before ingestion",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit number of documents to ingest",
    ),
) -> None:
    """Ingest documents from Paperless-ngx into Qdrant vector database."""
    import asyncio

    from vector_classifier.ingest import PaperlessIngestRunner

    console.print(
        "[bold cyan]Starting Paperless Document Ingestion into Qdrant...[/bold cyan]"
    )
    console.print(f"  Paperless URL: {url}")
    console.print(f"  Concurrency:   {concurrency}")
    console.print(f"  Batch size:    {batch_size}")
    console.print(f"  Recreate coll: {recreate}")
    if limit is not None:
        console.print(f"  Limit:         {limit}")

    runner = PaperlessIngestRunner(
        paperless_url=url,
        paperless_token=token,
        concurrency=concurrency,
        batch_size=batch_size,
        recreate_collection=recreate,
    )

    async def _execute() -> None:
        try:
            await runner.run(limit=limit, console=console)
        finally:
            await runner.close()

    asyncio.run(_execute())


@app.command()
def calibrate_tax(
    privat_file: str = typer.Option(
        "/home/gotti/paperless-custom-classifier/steuern-privat.txt",
        "--privat-file",
        help="Path to curated steuern-privat.txt",
    ),
    vermietung_file: str = typer.Option(
        "/home/gotti/paperless-custom-classifier/steuern-vermietung.txt",
        "--vermietung-file",
        help="Path to curated steuern-vermietung.txt",
    ),
    candidates_csv: str = typer.Option(
        "tax_candidates_2026.csv",
        "--candidates-csv",
        help="Output CSV path for extracted candidates",
    ),
    threshold_knn: float = typer.Option(
        0.82,
        "--threshold-knn",
        help="Minimum cosine similarity against curated tax anchors",
    ),
) -> None:
    """Calibrate Qdrant tax relevance across 3 temporal zones and export 2026+ candidates."""
    import asyncio
    from pathlib import Path

    from vector_classifier.calibrate import TaxCalibrationRunner

    runner = TaxCalibrationRunner(
        privat_file=Path(privat_file),
        vermietung_file=Path(vermietung_file),
        candidates_csv=Path(candidates_csv),
        threshold_knn=threshold_knn,
    )

    async def _execute() -> None:
        try:
            stats = await runner.run(console=console)
            console.print("\n[bold cyan]Summary of 3-Zone Calibration:[/bold cyan]")
            console.print(
                f"  Zone 2020-2025 (Curated): {stats['curated_2020_2025']} tax / {stats['nontax_2020_2025']} non-tax"
            )
            console.print(
                f"  Zone < 2020 (Historical): {stats['historical_pre_2020_tax']} tax / {stats['historical_pre_2020_nontax']} non-tax"
            )
            console.print(
                f"  Zone 2026+ (Candidates):  {stats['post_2025_tax_candidates']} candidates / {stats['post_2025_total']} total"
            )
        finally:
            await runner.close()

    asyncio.run(_execute())


@app.command()
def tag_candidates(
    candidates_csv: str = typer.Option(
        "tax_candidates_2026.csv",
        "--csv",
        help="Path to tax candidates CSV file",
    ),
    dry_run: bool = typer.Option(
        False,
        DRY_RUN_OPTION,
        help="Dry run without modifying Paperless tags",
    ),
) -> None:
    """Tag candidate documents in Paperless with 'Steuern-Review' and proposed tax category."""
    import asyncio
    from pathlib import Path

    from vector_classifier.tax_review import TaxReviewManager

    manager = TaxReviewManager()

    async def _execute() -> None:
        try:
            console.print(
                f"[bold cyan]Tagging candidates in Paperless (dry_run={dry_run})...[/bold cyan]"
            )
            stats = await manager.tag_candidates(
                csv_path=Path(candidates_csv),
                dry_run=dry_run,
                console=console,
            )
            console.print("\n[bold green]Candidate Tagging Complete:[/bold green]")
            console.print(f"  Candidates found: {stats['candidates']}")
            console.print(f"  Updated in Paperless: {stats['updated']}")
            console.print(f"  Already tagged: {stats['already_tagged']}")
            console.print(f"  Errors: {stats['errors']}")
        finally:
            await manager.close()

    asyncio.run(_execute())


@app.command()
def sync_curated(
    privat_file: str = typer.Option(
        "/home/gotti/paperless-custom-classifier/steuern-privat.txt",
        "--privat-file",
        help="Path to steuern-privat.txt",
    ),
    vermietung_file: str = typer.Option(
        "/home/gotti/paperless-custom-classifier/steuern-vermietung.txt",
        "--vermietung-file",
        help="Path to steuern-vermietung.txt",
    ),
    dry_run: bool = typer.Option(
        False,
        DRY_RUN_OPTION,
        help="Dry run without writing updates to Qdrant or text files",
    ),
) -> None:
    """Synchronize manually reviewed candidates from Paperless into Qdrant curated status."""
    import asyncio
    from pathlib import Path

    from vector_classifier.tax_review import TaxReviewManager

    manager = TaxReviewManager(
        privat_file=Path(privat_file),
        vermietung_file=Path(vermietung_file),
    )

    async def _execute() -> None:
        try:
            console.print(
                f"[bold cyan]Syncing reviewed candidates from Paperless (dry_run={dry_run})...[/bold cyan]"
            )
            stats = await manager.sync_curated(dry_run=dry_run, console=console)
            console.print(
                "\n[bold green]Curated Synchronization Complete:[/bold green]"
            )
            console.print(f"  Total Candidates Checked: {stats['total_candidates']}")
            console.print(f"  Confirmed (Curated Tax):  {stats['confirmed']}")
            console.print(f"  Rejected (Non-Tax):       {stats['rejected']}")
            console.print(f"  Still Pending Review:     {stats['pending']}")
            console.print(f"  Errors:                   {stats['errors']}")
        finally:
            await manager.close()

    asyncio.run(_execute())


if __name__ == "__main__":
    app()
