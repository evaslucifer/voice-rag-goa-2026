"""Tests for DATA_MODE configuration and DataProvider abstraction."""

import os
from unittest.mock import patch
import pytest

from app.config import Settings, get_settings
from ingestion.data_provider import DataProvider, get_data_provider


def test_default_data_mode() -> None:
    """Verify DATA_MODE defaults to 'demo'."""
    settings = Settings()
    assert settings.DATA_MODE == "demo"
    assert settings.is_demo_mode is True
    assert settings.QDRANT_COLLECTION == "msmarco_demo"


def test_production_data_mode_flag() -> None:
    """Verify is_demo_mode returns False when DATA_MODE is production."""
    with patch.dict(os.environ, {"DATA_MODE": "production", "QDRANT_COLLECTION": "msmarco_prod"}):
        settings = Settings()
        assert settings.DATA_MODE == "production"
        assert settings.is_demo_mode is False
        assert settings.QDRANT_COLLECTION == "msmarco_prod"


def test_data_provider_paths() -> None:
    """Verify DataProvider routes to demo files when in demo mode."""
    provider_demo = DataProvider(data_mode="demo")
    assert provider_demo.is_demo is True
    assert "demo_msmarco_xi.jsonl" in provider_demo.get_dataset_path()
    assert "demo_test_queries.json" in provider_demo.get_queries_path()
    assert provider_demo.get_target_collection() == "msmarco_demo"


def test_data_provider_load_demo_documents() -> None:
    """Verify DataProvider loads documents correctly from demo file."""
    provider = get_data_provider(data_mode="demo")
    docs = provider.load_documents(max_records=10)
    assert len(docs) > 0
    for doc in docs:
        assert doc.text
        assert doc.document_id
        assert doc.language in ("en", "hi", "mr")
