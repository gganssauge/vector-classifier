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


if __name__ == "__main__":
    app()
