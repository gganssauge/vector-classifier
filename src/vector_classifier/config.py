"""Application configuration settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama settings
    ollama_base_url: str = Field(
        default="http://ollama.ai.svc.cluster.local:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_embed_model: str = Field(
        default="bge-m3",
        validation_alias="OLLAMA_EMBED_MODEL",
    )
    ollama_timeout: float = Field(
        default=60.0,
        validation_alias="OLLAMA_TIMEOUT",
    )

    # Qdrant settings
    qdrant_host: str = Field(
        default="qdrant.ai.svc.cluster.local",
        validation_alias="QDRANT_HOST",
    )
    qdrant_port: int = Field(
        default=6333,
        validation_alias="QDRANT_PORT",
    )
    qdrant_collection: str = Field(
        default="paperless_documents",
        validation_alias="QDRANT_COLLECTION",
    )
    qdrant_api_key: str = Field(
        default="",
        validation_alias="QDRANT_API_KEY",
    )

    # Server settings
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    debug: bool = Field(default=False, validation_alias="DEBUG")

    # Classification Thresholds
    threshold_tax_confident: float = Field(
        default=0.82,
        validation_alias="THRESHOLD_TAX_CONFIDENT",
    )
    threshold_tax_review: float = Field(
        default=0.70,
        validation_alias="THRESHOLD_TAX_REVIEW",
    )
    threshold_tag: float = Field(
        default=0.78,
        validation_alias="THRESHOLD_TAG",
    )
    threshold_doctype_confident: float = Field(
        default=0.80,
        validation_alias="THRESHOLD_DOCTYPE_CONFIDENT",
    )


settings = Settings()
