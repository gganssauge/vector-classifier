"""FastAPI application entrypoint for vector-classifier service."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from rich.console import Console

from vector_classifier import __version__
from vector_classifier.config import settings
from vector_classifier.models.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
)
from vector_classifier.vectors.classifier import VectorClassifierEngine

console = Console()

_engine: VectorClassifierEngine | None = None


def get_engine() -> VectorClassifierEngine:
    """Return singleton VectorClassifierEngine or lazily instantiate."""
    global _engine
    if _engine is None:
        _engine = VectorClassifierEngine()
    return _engine


def set_engine(engine: VectorClassifierEngine | None) -> None:
    """Explicitly set or reset the VectorClassifierEngine instance (e.g. for testing)."""
    global _engine
    _engine = engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    console.print(
        f"[bold green]Starting vector-classifier v{__version__}[/bold green] "
        f"(Ollama: {settings.ollama_base_url}, Qdrant: {settings.qdrant_host}:{settings.qdrant_port})"
    )
    engine = get_engine()
    try:
        await engine.initialize_anchor_embeddings()
    except Exception as e:
        console.print(f"[yellow]Anchor pre-computation deferred: {e}[/yellow]")
    yield
    console.print("[yellow]Shutting down vector-classifier...[/yellow]")
    if _engine is not None:
        await _engine.close()


app = FastAPI(
    title="Vector Classifier Service",
    description="Microservice for semantic vector classification and search",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> HealthResponse:
    """Service health and connectivity probe."""
    engine = get_engine()
    ollama_ok = (
        await engine.ollama_client.health_check()
        if engine.ollama_client is not None
        else False
    )
    qdrant_ok = (
        await engine.qdrant_manager.health_check()
        if engine.qdrant_manager is not None
        else False
    )
    return HealthResponse(
        status="ok",
        version=__version__,
        ollama_connected=ollama_ok,
        qdrant_connected=qdrant_ok,
    )


@app.post("/api/v1/classify")
async def classify(payload: ClassifyRequest) -> ClassifyResponse:
    """Classify a document using vector representations."""
    engine = get_engine()
    result = await engine.classify(
        content=payload.content,
        filename=payload.filename,
        sha256=payload.sha256,
        header_content=payload.header_content,
    )
    return result
