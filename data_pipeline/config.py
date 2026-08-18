"""Data Pipeline Configuration Module.

Manages parameters for dataset downloading, cleaning, chunking strategies,
multilingual FastEmbed embeddings, Qdrant indexing, and 100-query benchmark evaluation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataPipelineConfig:
    """Configuration container for the Data Analyst pipeline."""

    # -------------------------------------------------------------------------
    # Base Directories
    # -------------------------------------------------------------------------
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    RAW_DATA_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "raw")
    PROCESSED_DATA_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "processed")
    RESULTS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "results")

    # -------------------------------------------------------------------------
    # Dataset Ingestion Parameters
    # -------------------------------------------------------------------------
    DATASET_NAME: str = "ai4bharat/MSMARCO-XI"
    SUPPORTED_LANGUAGES: List[str] = field(
        default_factory=lambda: ["en", "hi", "mr", "bn", "te", "ta", "hinglish"]
    )
    DEFAULT_LANGUAGE: str = "en"
    MAX_RECORDS_PER_LANG: int = 500
    RAW_DATASET_FILENAME: str = "msmarco_xi_raw.jsonl"
    PROCESSED_DATASET_FILENAME: str = "msmarco_xi_clean.jsonl"

    # -------------------------------------------------------------------------
    # Multilingual Embedding Model (FastEmbed)
    # Selected: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    # - Languages: 50+ languages (English, Hindi, Marathi, Bengali, Telugu, Tamil, etc.)
    # - Dimensions: 384 (Fast, lightweight ONNX CPU inference)
    # - Metric: Cosine Distance
    # -------------------------------------------------------------------------
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "DATA_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32
    DISTANCE_METRIC: str = "Cosine"

    # -------------------------------------------------------------------------
    # Qdrant Vector Database Configuration
    # -------------------------------------------------------------------------
    QDRANT_MODE: str = os.getenv("QDRANT_MODE", "local")  # "memory", "local", or "remote"
    QDRANT_PATH: str = os.getenv("QDRANT_PATH", "./qdrant_storage")
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL", None)
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION", "msmarco_multilingual_v1")
    QDRANT_BATCH_SIZE: int = 64
    QDRANT_TIMEOUT_SECONDS: float = 10.0

    # -------------------------------------------------------------------------
    # Chunking Strategy Defaults
    # -------------------------------------------------------------------------
    # Strategy A: Semantic Chunking
    SEMANTIC_TARGET_WORDS: int = 180
    SEMANTIC_MAX_WORDS: int = 240
    # Strategy B: Hierarchical Chunking
    HIERARCHICAL_PARENT_WORDS: int = 512
    HIERARCHICAL_CHILD_WORDS: int = 128
    HIERARCHICAL_CHILD_OVERLAP: int = 20
    # Strategy C: Overlap-based Chunking
    OVERLAP_CHUNK_WORDS: int = 200
    OVERLAP_CHUNK_OVERLAP: int = 40

    # -------------------------------------------------------------------------
    # 100-Query Benchmark Suite Parameters
    # Categories: exactly 5 categories totalling 100 queries
    # - in_domain: 50
    # - ambiguous: 15
    # - no_answer: 15
    # - off_topic: 10
    # - prompt_injection: 10
    # -------------------------------------------------------------------------
    BENCHMARK_QUERIES_FILE: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "data" / "benchmark_100_queries.json"
    )
    BENCHMARK_WARMUP_RUNS: int = 5
    BENCHMARK_TOP_K: int = 5
    BENCHMARK_SCORE_THRESHOLD: float = 0.50

    def __post_init__(self) -> None:
        """Ensure runtime directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_pipeline_config() -> DataPipelineConfig:
    """Factory helper to obtain data pipeline configuration."""
    return DataPipelineConfig()
