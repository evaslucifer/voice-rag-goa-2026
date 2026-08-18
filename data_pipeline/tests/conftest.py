"""Pytest fixtures for data pipeline unit and integration tests."""

import pytest
from qdrant_client import QdrantClient

from data_pipeline.chunking_strategies import ChunkRecord
from data_pipeline.config import DataPipelineConfig
from data_pipeline.dataset_downloader import NormalizedDocument
from data_pipeline.vector_indexer import VectorIndexer


@pytest.fixture
def mock_config(tmp_path) -> DataPipelineConfig:
    """Fixture providing temporary isolated paths for testing."""
    config = DataPipelineConfig()
    config.DATA_DIR = tmp_path / "data"
    config.RAW_DATA_DIR = tmp_path / "data" / "raw"
    config.PROCESSED_DATA_DIR = tmp_path / "data" / "processed"
    config.RESULTS_DIR = tmp_path / "data" / "results"
    config.QDRANT_MODE = "memory"
    config.QDRANT_COLLECTION_NAME = "test_collection"
    config.__post_init__()
    return config


@pytest.fixture
def sample_normalized_documents() -> list[NormalizedDocument]:
    """Fixture providing sample multilingual normalized documents."""
    return [
        NormalizedDocument(
            document_id="doc_en_01",
            title="The Manhattan Project",
            text="The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons. It was led by the United States with support from the UK and Canada. J. Robert Oppenheimer led the Los Alamos Laboratory.",
            language="en",
            source="ai4bharat/MSMARCO-XI",
            query="What was the Manhattan Project?",
            answers=["The Manhattan Project was a research undertaking that developed nuclear weapons."],
        ),
        NormalizedDocument(
            document_id="doc_hi_01",
            title="भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो)",
            text="भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) भारत की राष्ट्रीय अंतरिक्ष एजेंसी है। इसकी स्थापना 15 अगस्त 1969 को डॉ. विक्रम साराभाई के नेतृत्व में की गई थी। इसका मुख्यालय बेंगलुरु में स्थित है।",
            language="hi",
            source="ai4bharat/MSMARCO-XI",
            query="इसरो की स्थापना कब हुई थी?",
            answers=["इसरो की स्थापना 15 अगस्त 1969 को हुई थी।"],
        ),
        NormalizedDocument(
            document_id="doc_mr_01",
            title="मराठा साम्राज्य",
            text="छत्रपती शिवाजी महाराज यांनी १७ व्या शतकात मराठा साम्राज्याची स्थापना केली। त्यांनी गनिमी कावा युद्धनीती वापरली। रायगड किल्ला ही त्यांची राजधानी होती।",
            language="mr",
            source="ai4bharat/MSMARCO-XI",
            query="मराठा साम्राज्याची राजधानी कोणती होती?",
            answers=["रायगड किल्ला ही मराठा साम्राज्याची राजधानी होती."],
        ),
    ]


@pytest.fixture
def sample_chunk_records() -> list[ChunkRecord]:
    """Fixture providing sample chunk records."""
    return [
        ChunkRecord(
            chunk_id="chk_001",
            document_id="doc_en_01",
            text="The Manhattan Project produced the first nuclear weapons during World War II.",
            strategy="semantic",
            position=0,
            language="en",
            source="ai4bharat/MSMARCO-XI",
        ),
        ChunkRecord(
            chunk_id="chk_002",
            document_id="doc_hi_01",
            text="इसरो की स्थापना 15 अगस्त 1969 को डॉ. विक्रम साराभाई के नेतृत्व में हुई थी।",
            strategy="semantic",
            position=0,
            language="hi",
            source="ai4bharat/MSMARCO-XI",
        ),
        ChunkRecord(
            chunk_id="chk_003",
            document_id="doc_mr_01",
            text="छत्रपती शिवाजी महाराजांची राजधानी रायगड किल्ला होती.",
            strategy="semantic",
            position=0,
            language="mr",
            source="ai4bharat/MSMARCO-XI",
        ),
    ]


@pytest.fixture
def in_memory_indexer(mock_config) -> VectorIndexer:
    """Fixture providing an isolated in-memory VectorIndexer."""
    return VectorIndexer(config=mock_config, mode="memory")
