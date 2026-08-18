"""Unit tests for dataset downloading, cleaning, and normalization."""

import json
from pathlib import Path
import pytest

from data_pipeline.dataset_downloader import (
    DataCleanerAndNormalizer,
    MSMARCODatasetDownloader,
    NormalizedDocument,
)


def test_data_cleaner_unicode_and_whitespace():
    """Test that unicode NFKC normalization, whitespace stripping, and Indic characters work correctly."""
    cleaner = DataCleanerAndNormalizer()

    dirty_text = "  Hello \t\t World! \n\n\n This is   a test. \x00\x05  "
    clean = cleaner.clean_text(dirty_text)
    assert clean == "Hello World!\n\nThis is a test."

    indic_text = "  इसरो की स्थापना   1969 में हुई।  "
    clean_indic = cleaner.clean_text(indic_text)
    assert clean_indic == "इसरो की स्थापना 1969 में हुई।"


def test_normalize_record_valid():
    """Test normalizing a standard valid record with query and answers."""
    cleaner = DataCleanerAndNormalizer()
    raw = {
        "id": "doc_100",
        "title": "Quantum Mechanics",
        "text": "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature.",
        "language": "en",
        "query": "What is quantum mechanics?",
        "answers": ["Quantum mechanics describes the fundamental properties of nature."],
    }

    doc = cleaner.normalize_record(raw)
    assert doc is not None
    assert doc.document_id == "doc_100"
    assert doc.title == "Quantum Mechanics"
    assert doc.language == "en"
    assert doc.query == "What is quantum mechanics?"
    assert len(doc.answers) == 1
    assert "fundamental theory" in doc.text


def test_normalize_record_nested_passages():
    """Test normalizing records with nested passages list format."""
    cleaner = DataCleanerAndNormalizer()
    raw = {
        "query_id": "q_nested_01",
        "passages": [
            {
                "passage_id": "p_nested_99",
                "text": "Deep learning uses multiple layers of artificial neural networks.",
                "is_selected": True,
            }
        ],
        "language": "hin",
    }

    doc = cleaner.normalize_record(raw)
    assert doc is not None
    assert doc.document_id == "p_nested_99"
    assert "Deep learning" in doc.text
    assert doc.language == "hi"  # Normalized from 'hin' to 'hi'


def test_normalize_record_empty_or_too_short():
    """Test that empty or extremely short/corrupted inputs are safely rejected."""
    cleaner = DataCleanerAndNormalizer()

    assert cleaner.normalize_record({}) is None
    assert cleaner.normalize_record({"text": ""}) is None
    assert cleaner.normalize_record({"text": "short"}) is None  # Below min_char_length 15
    assert cleaner.normalize_record("not a dict") is None


def test_deduplication():
    """Test that duplicate records with identical content are filtered out."""
    cleaner = DataCleanerAndNormalizer()
    record1 = {"id": "doc_1", "text": "The quick brown fox jumps over the lazy dog."}
    record2 = {"id": "doc_2", "text": "  THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG.  "}

    doc1 = cleaner.normalize_record(record1, deduplicate=True)
    doc2 = cleaner.normalize_record(record2, deduplicate=True)

    assert doc1 is not None
    assert doc2 is None  # Duplicate content filtered


def test_save_and_load_jsonl_roundtrip(tmp_path):
    """Test saving normalized documents to JSONL and reading back."""
    downloader = MSMARCODatasetDownloader()
    docs = [
        NormalizedDocument(document_id="d1", text="First test document content.", language="en"),
        NormalizedDocument(document_id="d2", text="Second test document content.", language="hi"),
    ]

    out_file = tmp_path / "test_output.jsonl"
    downloader.save_to_jsonl(docs, out_file)
    assert out_file.exists()

    loaded = downloader.load_local_jsonl(out_file)
    assert len(loaded) == 2
    assert loaded[0].document_id == "d1"
    assert loaded[1].document_id == "d2"
    assert loaded[1].language == "hi"


def test_generate_representative_multilingual_corpus():
    """Test generation of multilingual fallback corpus across all target languages."""
    downloader = MSMARCODatasetDownloader()
    languages = ["en", "hi", "mr", "bn", "te", "ta", "hinglish"]
    docs = downloader.generate_representative_multilingual_corpus(languages=languages, records_per_lang=2)

    assert len(docs) == len(languages) * 2
    found_langs = {d.language for d in docs}
    assert "en" in found_langs
    assert "hi" in found_langs
    assert "mr" in found_langs
    assert "hinglish" in found_langs
