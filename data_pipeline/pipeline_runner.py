"""Master Pipeline Runner for Data Analyst MSMARCO-XI Ingestion and Benchmarking.

Chains the complete workflow:
Dataset Downloader -> Cleaning & Normalization -> Chunking Strategies -> FastEmbed Indexing -> 100-Query Benchmark Suite.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_pipeline.benchmark_suite import BenchmarkSuite
from data_pipeline.chunking_strategies import ChunkRecord, get_chunker
from data_pipeline.config import DataPipelineConfig, get_pipeline_config
from data_pipeline.dataset_downloader import MSMARCODatasetDownloader, NormalizedDocument
from data_pipeline.vector_indexer import VectorIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("data_pipeline.pipeline_runner")


class DataPipelineRunner:
    """Master orchestrator connecting all data pipeline stages."""

    def __init__(self, config: Optional[DataPipelineConfig] = None) -> None:
        self.config = config or get_pipeline_config()
        self.downloader = MSMARCODatasetDownloader(config=self.config)
        self.indexer: Optional[VectorIndexer] = None

    def run_download_and_clean(
        self,
        languages: Optional[List[str]] = None,
        max_records_per_lang: int = 50,
        local_raw_path: Optional[Path] = None,
    ) -> List[NormalizedDocument]:
        """Stage 1: Download, clean, normalize, and load documents."""
        logger.info("\n" + "=" * 70)
        logger.info("  STAGE 1: DATASET DOWNLOAD & CLEANING")
        logger.info("=" * 70)

        clean_file = self.downloader.run_pipeline(
            languages=languages,
            max_records_per_lang=max_records_per_lang,
            local_raw_path=local_raw_path,
        )
        docs = self.downloader.load_local_jsonl(clean_file)
        logger.info("Stage 1 completed: %d normalized documents loaded.", len(docs))
        return docs

    def run_chunking(
        self,
        documents: List[NormalizedDocument],
        strategy: str = "semantic",
    ) -> List[ChunkRecord]:
        """Stage 2: Chunk normalized documents using specified strategy."""
        logger.info("\n" + "=" * 70)
        logger.info("  STAGE 2: CHUNKING (Strategy: '%s')", strategy)
        logger.info("=" * 70)

        chunker = get_chunker(strategy)
        chunks = chunker.chunk_documents(documents)
        logger.info("Stage 2 completed: Generated %d chunks from %d documents.", len(chunks), len(documents))
        return chunks

    def run_indexing(
        self,
        chunks: List[ChunkRecord],
        recreate_collection: bool = True,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stage 3: Generate FastEmbed embeddings and index into Qdrant."""
        logger.info("\n" + "=" * 70)
        logger.info("  STAGE 3: FASTEMBED ONNX EMBEDDING & QDRANT INDEXING")
        logger.info("  Model: %s | Dimension: %d | Mode: %s", self.config.EMBEDDING_MODEL_NAME, self.config.EMBEDDING_DIMENSION, mode or self.config.QDRANT_MODE)
        logger.info("=" * 70)

        if self.indexer is None:
            self.indexer = VectorIndexer(config=self.config, mode=mode)

        stats = self.indexer.index_chunks(
            chunks=chunks,
            recreate_collection=recreate_collection,
        )
        logger.info("Stage 3 completed: Indexed %d points in %.2f ms.", stats["indexed_count"], stats["latency_ms"])
        return stats

    def run_benchmark(
        self,
        queries_file: Optional[Path] = None,
        warmup: bool = True,
    ) -> Dict[str, Any]:
        """Stage 4: Run 100-query benchmark suite and generate reports."""
        logger.info("\n" + "=" * 70)
        logger.info("  STAGE 4: 100-QUERY BENCHMARK & LATENCY PROFILING")
        logger.info("=" * 70)

        if self.indexer is None:
            self.indexer = VectorIndexer(config=self.config)

        suite = BenchmarkSuite(
            indexer=self.indexer,
            config=self.config,
            queries_file=queries_file,
        )
        report, results = suite.run_benchmark(warmup=warmup)
        suite.print_console_summary(report)
        json_path, csv_path = suite.export_results(report, results)

        return {
            "report": report.to_dict(),
            "json_path": str(json_path),
            "csv_path": str(csv_path),
        }

    def run_end_to_end(
        self,
        languages: Optional[List[str]] = None,
        max_records_per_lang: int = 25,
        strategy: str = "semantic",
        mode: Optional[str] = None,
        local_raw_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute full end-to-end pipeline."""
        t0 = time.perf_counter()
        logger.info("\n" + "#" * 70)
        logger.info("  STARTING END-TO-END DATA ANALYST PIPELINE EXECUTION")
        logger.info("#" * 70)

        # 1. Download & Clean
        docs = self.run_download_and_clean(
            languages=languages,
            max_records_per_lang=max_records_per_lang,
            local_raw_path=local_raw_path,
        )

        # 2. Chunk
        chunks = self.run_chunking(docs, strategy=strategy)

        # 3. Index
        index_stats = self.run_indexing(chunks, recreate_collection=True, mode=mode)

        # 4. Benchmark
        benchmark_stats = self.run_benchmark(warmup=True)

        total_e2e_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logger.info("\n" + "#" * 70)
        logger.info("  END-TO-END PIPELINE COMPLETED IN %.2f ms (%.2f s)", total_e2e_ms, total_e2e_ms / 1000.0)
        logger.info("#" * 70)

        return {
            "total_documents": len(docs),
            "total_chunks": len(chunks),
            "index_stats": index_stats,
            "benchmark_stats": benchmark_stats,
            "total_time_ms": total_e2e_ms,
        }


def main() -> None:
    """CLI Interface for Data Pipeline Runner."""
    parser = argparse.ArgumentParser(description="Data Pipeline Master Runner.")
    parser.add_argument("--all", action="store_true", help="Run complete end-to-end pipeline")
    parser.add_argument("--download", action="store_true", help="Run dataset download and cleaning")
    parser.add_argument("--chunk", action="store_true", help="Run chunking on cleaned dataset")
    parser.add_argument("--index", action="store_true", help="Run FastEmbed indexing into Qdrant")
    parser.add_argument("--benchmark", action="store_true", help="Run 100-query benchmark suite")
    parser.add_argument("--strategy", type=str, default="semantic", choices=["semantic", "hierarchical", "overlap"], help="Chunking strategy")
    parser.add_argument("--mode", type=str, default="local", choices=["memory", "local", "remote"], help="Qdrant mode")
    parser.add_argument("--max-records", type=int, default=25, help="Max records per language")
    parser.add_argument("--input-file", type=str, default=None, help="Optional raw JSONL file")
    args = parser.parse_args()

    runner = DataPipelineRunner()

    if args.all or (not args.download and not args.chunk and not args.index and not args.benchmark):
        runner.run_end_to_end(
            max_records_per_lang=args.max_records,
            strategy=args.strategy,
            mode=args.mode,
            local_raw_path=Path(args.input_file) if args.input_file else None,
        )
    else:
        docs: List[NormalizedDocument] = []
        chunks: List[ChunkRecord] = []

        if args.download:
            docs = runner.run_download_and_clean(
                max_records_per_lang=args.max_records,
                local_raw_path=Path(args.input_file) if args.input_file else None,
            )

        if args.chunk:
            if not docs:
                clean_path = runner.config.PROCESSED_DATA_DIR / runner.config.PROCESSED_DATASET_FILENAME
                if clean_path.exists():
                    docs = runner.downloader.load_local_jsonl(clean_path)
                else:
                    docs = runner.run_download_and_clean(max_records_per_lang=args.max_records)
            chunks = runner.run_chunking(docs, strategy=args.strategy)

        if args.index:
            if not chunks:
                if not docs:
                    clean_path = runner.config.PROCESSED_DATA_DIR / runner.config.PROCESSED_DATASET_FILENAME
                    if clean_path.exists():
                        docs = runner.downloader.load_local_jsonl(clean_path)
                    else:
                        docs = runner.run_download_and_clean(max_records_per_lang=args.max_records)
                chunks = runner.run_chunking(docs, strategy=args.strategy)
            runner.run_indexing(chunks, mode=args.mode)

        if args.benchmark:
            runner.run_benchmark(warmup=True)


if __name__ == "__main__":
    main()
