"""Tests for MSMARCO-XI dataset loader and normalizer."""

import os
import tempfile
import json
import pytest
from ingestion.dataset_loader import DatasetLoader, NormalizedDocument


def test_normalize_record_valid() -> None:
    """Test normalization of valid record with varied key naming."""
    loader = DatasetLoader()
    raw = {
        "passage_id": "p_101",
        "passage": "Quantum mechanics is a fundamental theory in physics.",
        "title": "Quantum Physics",
        "language": "en",
        "query": "What is quantum mechanics?",
        "answers": ["A fundamental theory in physics."],
    }
    doc = loader.normalize_record(raw)
    assert doc is not None
    assert isinstance(doc, NormalizedDocument)
    assert doc.document_id == "p_101"
    assert doc.title == "Quantum Physics"
    assert doc.language == "en"
    assert doc.query == "What is quantum mechanics?"
    assert len(doc.answers) == 1


def test_normalize_record_empty_or_invalid() -> None:
    """Test empty or malformed records are filtered out."""
    loader = DatasetLoader()
    assert loader.normalize_record({}) is None
    assert loader.normalize_record({"passage": ""}) is None
    assert loader.normalize_record("not a dict") is None


def test_load_from_jsonl() -> None:
    """Test loading and parsing records from temporary JSONL file."""
    loader = DatasetLoader()
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps({"id": "d1", "text": "First passage content.", "language": "en"}) + "\n")
        f.write("\n")  # Empty line
        f.write(json.dumps({"id": "d2", "text": "Second passage content.", "language": "hi"}) + "\n")
        temp_path = f.name

    try:
        docs = loader.load_from_jsonl(temp_path)
        assert len(docs) == 2
        assert docs[0].document_id == "d1"
        assert docs[1].document_id == "d2"
        assert docs[1].language == "hi"
    finally:
        os.remove(temp_path)
