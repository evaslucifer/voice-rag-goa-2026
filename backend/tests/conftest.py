"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient
from app.config import Settings, get_settings
from app.main import app


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing customized test settings."""
    return Settings(
        ENVIRONMENT="test",
        LOG_LEVEL="DEBUG",
        SARVAM_API_KEY="test_sarvam_key",
        GROQ_API_KEY="test_groq_key",
        GEMINI_API_KEY="test_gemini_key",
        QDRANT_URL="http://localhost:6333",
        QDRANT_COLLECTION="test_collection",
    )


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    """Fixture providing FastAPI TestClient with overridden settings."""
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
