"""Data Pipeline Package for Multilingual MSMARCO-XI Voice-RAG System.

Modules:
- dataset_downloader: Downloading, cleaning, normalizing, and deduplicating multilingual data.
- chunking_strategies: Semantic, Hierarchical (Parent-Child), and Overlap chunkers.
- vector_indexer: FastEmbed multilingual embeddings (384-dim) and Qdrant indexing.
- benchmark_suite: 100-query benchmark evaluating P50, P70, P100 latency across 5 categories.
- pipeline_runner: End-to-end master orchestrator.
- config: Configuration parameters for the data pipeline.
"""

from data_pipeline.benchmark_suite import BenchmarkQuery, BenchmarkReport, BenchmarkSuite
from data_pipeline.chunking_strategies import (
    BaseChunker,
    ChunkRecord,
    HierarchicalChunker,
    OverlapChunker,
    SemanticChunker,
    get_chunker,
)
from data_pipeline.config import DataPipelineConfig, get_pipeline_config
from data_pipeline.dataset_downloader import (
    DataCleanerAndNormalizer,
    MSMARCODatasetDownloader,
    NormalizedDocument,
)
from data_pipeline.pipeline_runner import DataPipelineRunner
from data_pipeline.vector_indexer import SearchResult, VectorIndexer

__all__ = [
    "DataPipelineConfig",
    "get_pipeline_config",
    "NormalizedDocument",
    "DataCleanerAndNormalizer",
    "MSMARCODatasetDownloader",
    "ChunkRecord",
    "BaseChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    "OverlapChunker",
    "get_chunker",
    "SearchResult",
    "VectorIndexer",
    "BenchmarkQuery",
    "BenchmarkReport",
    "BenchmarkSuite",
    "DataPipelineRunner",
]
