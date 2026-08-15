"""Application configuration management using Pydantic Settings."""

import json
from functools import lru_cache
from typing import Any, List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env", "../.env", "../../.env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    DATA_MODE: str = Field(default="demo", description="Data mode: 'demo' (lightweight curated demo data) or 'production' (full MSMARCO-XI)")
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "*"]
    FRONTEND_ORIGIN: Optional[str] = Field(default=None, description="Deployed frontend URL (e.g. https://my-frontend.vercel.app)")

    # External API Keys
    SARVAM_API_KEY: Optional[str] = Field(default=None, description="Sarvam AI API key for STT")
    GROQ_API_KEY: Optional[str] = Field(default=None, description="Groq API key for Llama 3.1 LLM inference")
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Gemini API key for fallback inference")

    # Qdrant Vector Database
    QDRANT_URL: Optional[str] = Field(default=None, description="Optional Qdrant server instance URL")
    QDRANT_PATH: str = Field(default="./qdrant_storage", description="Local embedded Qdrant storage path")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Optional Qdrant Cloud API key")
    QDRANT_COLLECTION: str = Field(default="msmarco_demo", description="Qdrant target collection name (default: msmarco_demo in demo mode)")
    QDRANT_TIMEOUT_SECONDS: float = Field(default=5.0, description="Qdrant client timeout in seconds")

    # Embedding Service Configuration (Local FastEmbed ONNX)
    EMBEDDING_MODEL_NAME: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="FastEmbed model name for query embedding"
    )
    EMBEDDING_BATCH_SIZE: int = Field(default=32, description="Batch size for embedding generation")

    # Retrieval Configuration (Target Top-K=3, Score Threshold=0.65)
    RETRIEVAL_TOP_K: int = Field(default=3, description="Number of top candidates to retrieve")
    RETRIEVAL_SCORE_THRESHOLD: float = Field(default=0.65, description="Minimum cosine similarity threshold")

    # Chunking Configuration
    CHUNKING_STRATEGY: str = Field(default="semantic", description="Default chunking strategy: semantic, parent_child, metadata, or fixed")
    CHUNK_SIZE: int = Field(default=200, description="Fixed chunk size in words")
    CHUNK_OVERLAP: int = Field(default=40, description="Fixed chunk overlap in words")
    PARENT_CHUNK_SIZE: int = Field(default=512, description="Parent chunk size in words for parent-child strategy")
    CHILD_CHUNK_SIZE: int = Field(default=128, description="Child chunk size in words for parent-child strategy")

    # LLM Service Configuration
    LLM_PROVIDER: str = Field(default="groq", description="LLM provider: groq, gemini, or local")
    LLM_MODEL: str = Field(default="llama-3.1-8b-instant", description="Primary LLM model identifier")
    LLM_FALLBACK_MODEL: str = Field(default="gemini-1.5-flash", description="Fallback LLM model identifier")
    LLM_TEMPERATURE: float = Field(default=0.1, description="LLM sampling temperature")
    LLM_TIMEOUT_SECONDS: float = Field(default=6.0, description="LLM timeout in seconds")

    # Guardrails
    GUARDRAILS_ENABLED: bool = Field(default=True, description="Enable safety and grounding guardrails")

    # Sarvam STT Service Configuration
    SARVAM_STT_LANGUAGE_CODE: str = Field(default="en-IN", description="Default BCP-47 language code for Sarvam STT")
    SARVAM_STT_MODEL: str = Field(default="saaras:v2", description="Sarvam STT model identifier")
    SARVAM_STT_TIMEOUT_SECONDS: float = Field(default=10.0, description="Sarvam API timeout in seconds")

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return ["*"]

    @property
    def effective_cors_origins(self) -> List[str]:
        """Return CORS origins list including FRONTEND_ORIGIN if specified."""
        origins = list(self.CORS_ORIGINS)
        if self.FRONTEND_ORIGIN and self.FRONTEND_ORIGIN not in origins:
            origins.append(self.FRONTEND_ORIGIN)
        return origins

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() in ("production", "prod")

    @property
    def is_demo_mode(self) -> bool:
        """Check if data mode is set to demo."""
        return self.DATA_MODE.lower() in ("demo", "test", "dev")

    @property
    def has_sarvam_key(self) -> bool:
        """Check if a valid Sarvam API key is configured."""
        return bool(self.SARVAM_API_KEY and not self.SARVAM_API_KEY.startswith("your_"))

    @property
    def has_groq_key(self) -> bool:
        """Check if a valid Groq API key is configured."""
        return bool(self.GROQ_API_KEY and not self.GROQ_API_KEY.startswith("your_"))

    @property
    def has_gemini_key(self) -> bool:
        """Check if a valid Gemini API key is configured."""
        return bool(self.GEMINI_API_KEY and not self.GEMINI_API_KEY.startswith("your_"))


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
