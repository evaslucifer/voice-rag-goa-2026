"""Tests for environment configuration and Pydantic settings."""

import os
from app.config import Settings, get_settings


def test_default_settings() -> None:
    """Test default values in Settings."""
    settings = Settings()
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8000
    assert settings.ENVIRONMENT == "development"
    assert settings.EMBEDDING_MODEL_NAME == "BAAI/bge-small-en-v1.5"
    assert settings.QDRANT_COLLECTION in ("msmarco_demo", "msmarco_xi_bge_small")
    assert not settings.is_production


def test_cors_origins_parsing() -> None:
    """Test CORS_ORIGINS validator parses string and list."""
    settings_str = Settings(CORS_ORIGINS="http://localhost:3000, http://127.0.0.1:3000")
    assert "http://localhost:3000" in settings_str.CORS_ORIGINS
    assert "http://127.0.0.1:3000" in settings_str.CORS_ORIGINS

    settings_list = Settings(CORS_ORIGINS=["http://example.com"])
    assert settings_list.CORS_ORIGINS == ["http://example.com"]


def test_api_key_helper_properties() -> None:
    """Test property helpers for checking valid API keys."""
    settings_empty = Settings(SARVAM_API_KEY=None, GROQ_API_KEY="", GEMINI_API_KEY="your_gemini_key")
    assert not settings_empty.has_sarvam_key
    assert not settings_empty.has_groq_key
    assert not settings_empty.has_gemini_key

    settings_valid = Settings(SARVAM_API_KEY="sk_sarvam_live_123", GROQ_API_KEY="gsk_12345", GEMINI_API_KEY="AIzaSy123")
    assert settings_valid.has_sarvam_key
    assert settings_valid.has_groq_key
    assert settings_valid.has_gemini_key


def test_get_settings_cached() -> None:
    """Test get_settings returns consistent cached singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
