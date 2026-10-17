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
        "--dry-run/--no-dry-run",
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


if __name__ == "__main__":
    app()
