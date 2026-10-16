"""FastAPI application entrypoint for vector-classifier service."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from rich.console import Console

from vector_classifier import __version__
from vector_classifier.config import settings
from vector_classifier.models.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
)

console = Console()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    console.print(
        f"[bold green]Starting vector-classifier v{__version__}[/bold green] "
        f"(Ollama: {settings.ollama_base_url}, Qdrant: {settings.qdrant_host}:{settings.qdrant_port})"
    )
    yield
    console.print("[yellow]Shutting down vector-classifier...[/yellow]")


app = FastAPI(
    title="Vector Classifier Service",
    description="Microservice for semantic vector classification and search",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Service health and connectivity probe."""
    return HealthResponse(
        status="ok",
        version=__version__,
        ollama_connected=False,
        qdrant_connected=False,
    )


@app.post("/api/v1/classify", response_model=ClassifyResponse)
async def classify(payload: ClassifyRequest) -> ClassifyResponse:
    """Classify a document using vector representations."""
    # Stub response until classifier engine is integrated in issue 04
    return ClassifyResponse(
        predicted_doc_type=None,
        doc_type_confidence=0.0,
        predicted_tags=[],
        tag_scores={},
        is_tax_relevant=False,
        tax_confidence=0.0,
        tax_category=None,
        tax_review_required=False,
    )
