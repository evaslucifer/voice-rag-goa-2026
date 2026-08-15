"""Tests for demo dataset structure and normalization."""

import os
import pytest
from ingestion.data_provider import DEMO_DATA_FILE
from ingestion.dataset_loader import DatasetLoader, load_from_jsonl


def test_demo_dataset_file_exists() -> None:
    """Verify demo dataset file is created and non-empty."""
    assert os.path.exists(DEMO_DATA_FILE)
    assert os.path.getsize(DEMO_DATA_FILE) > 500


def test_load_all_demo_records() -> None:
    """Verify all valid document records load and validate schema."""
    loader = DatasetLoader()
    docs = loader.load_from_jsonl(DEMO_DATA_FILE)
    assert len(docs) >= 20

    languages = set(d.language for d in docs)
    assert "en" in languages
    assert "hi" in languages
    assert "mr" in languages

    for doc in docs:
        assert doc.text
        assert doc.document_id
        assert doc.metadata.get("data_mode") == "DEMO DATA — NOT FINAL MSMARCO-XI DATA"


def test_nested_passages_normalization() -> None:
    """Verify normalize_record correctly extracts nested passages list."""
    loader = DatasetLoader()
    raw = {
        "id": "demo_test_01",
        "language": "hi",
        "query": "प्रकाश संश्लेषण क्या है?",
        "title": "पादप जीवविज्ञान",
        "category": "in_domain",
        "data_mode": "DEMO DATA — NOT FINAL MSMARCO-XI DATA",
        "passages": [
          {
            "passage_id": "p_hi_01",
            "text": "प्रकाश संश्लेषण द्वारा हरे पौधे सूर्य के प्रकाश से भोजन बनाते हैं।",
            "is_selected": True
          }
        ]
    }
    doc = loader.normalize_record(raw)
    assert doc is not None
    assert doc.document_id == "p_hi_01"
    assert "प्रकाश संश्लेषण" in doc.text
    assert doc.metadata["query_id"] == "demo_test_01"
    assert doc.metadata["is_selected"] is True
    assert doc.metadata["data_mode"] == "DEMO DATA — NOT FINAL MSMARCO-XI DATA"
