"""100-Query Benchmark Suite for Multilingual Voice-RAG Retrieval Pipeline.

Evaluates latency (P50, P70, P100, min, max, mean, stddev) and quality
across exact 5 categories (100 queries) with warm-up and structured CSV/JSON reporting.
"""

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_pipeline.config import DataPipelineConfig, get_pipeline_config
from data_pipeline.vector_indexer import SearchResult, VectorIndexer

logger = logging.getLogger("data_pipeline.benchmark_suite")

# Expected category counts for exact 100-query benchmark
EXPECTED_CATEGORY_COUNTS: Dict[str, int] = {
    "in_domain": 50,
    "ambiguous": 15,
    "no_answer": 15,
    "off_topic": 10,
    "prompt_injection": 10,
}
EXPECTED_TOTAL_QUERIES = 100


# =============================================================================
# Benchmark Query & Result Schemas
# =============================================================================
@dataclass
class BenchmarkQuery:
    """Benchmark query specification."""

    id: str
    query: str
    language: str
    category: str
    expected_status: str = "SUCCESS"
    keywords: List[str] = field(default_factory=list)


@dataclass
class QueryExecutionResult:
    """Individual query benchmark run metrics."""

    query_id: str
    query: str
    language: str
    category: str
    expected_status: str
    success: bool
    embedding_latency_ms: float
    search_latency_ms: float
    total_latency_ms: float
    hits_count: int
    top_score: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LatencyPercentiles:
    """Statistical summary of latency metrics."""

    count: int
    p50_ms: float
    p70_ms: float
    p100_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    stddev_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark execution report."""

    total_queries: int
    successful_queries: int
    failed_queries: int
    success_rate_percent: float
    category_counts: Dict[str, int]
    language_counts: Dict[str, int]
    total_latency: LatencyPercentiles
    embedding_latency: LatencyPercentiles
    search_latency: LatencyPercentiles
    category_breakdown: Dict[str, Dict[str, Any]]
    model_name: str
    embedding_dimension: int
    distance_metric: str
    collection_name: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate_percent": self.success_rate_percent,
            "category_counts": self.category_counts,
            "language_counts": self.language_counts,
            "total_latency": self.total_latency.to_dict(),
            "embedding_latency": self.embedding_latency.to_dict(),
            "search_latency": self.search_latency.to_dict(),
            "category_breakdown": self.category_breakdown,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "distance_metric": self.distance_metric,
            "collection_name": self.collection_name,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Latency Math Helper
# =============================================================================
def calculate_percentiles(latencies: List[float]) -> LatencyPercentiles:
    """Calculate P50, P70, P100, min, max, mean, and stddev from latencies in ms."""
    if not latencies:
        return LatencyPercentiles(
            count=0, p50_ms=0.0, p70_ms=0.0, p100_ms=0.0,
            min_ms=0.0, max_ms=0.0, mean_ms=0.0, stddev_ms=0.0
        )

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    def get_p(p: float) -> float:
        """Nearest rank percentile computation."""
        idx = int(math.ceil((p / 100.0) * n)) - 1
        return round(sorted_lat[max(0, min(idx, n - 1))], 2)

    p50 = get_p(50.0)
    p70 = get_p(70.0)
    p100 = round(sorted_lat[-1], 2)
    min_val = round(sorted_lat[0], 2)
    max_val = round(sorted_lat[-1], 2)
    mean_val = round(sum(sorted_lat) / n, 2)
    variance = sum((x - mean_val) ** 2 for x in sorted_lat) / n
    stddev_val = round(math.sqrt(variance), 2)

    return LatencyPercentiles(
        count=n,
        p50_ms=p50,
        p70_ms=p70,
        p100_ms=p100,
        min_ms=min_val,
        max_ms=max_val,
        mean_ms=mean_val,
        stddev_ms=stddev_val,
    )


# =============================================================================
# Benchmark Suite
# =============================================================================
class BenchmarkSuite:
    """Executes and analyzes 100-query benchmark runs on the vector indexer."""

    def __init__(
        self,
        indexer: VectorIndexer,
        config: Optional[DataPipelineConfig] = None,
        queries_file: Optional[Path] = None,
    ) -> None:
        self.indexer = indexer
        self.config = config or get_pipeline_config()
        self.queries_file = queries_file or self.config.BENCHMARK_QUERIES_FILE
        self.queries: List[BenchmarkQuery] = self.load_and_validate_queries(self.queries_file)

    def load_and_validate_queries(self, file_path: Path) -> List[BenchmarkQuery]:
        """Load 100 queries from JSON and strictly validate category counts."""
        if not file_path.exists():
            raise FileNotFoundError(f"Benchmark queries file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError(f"Benchmark file {file_path} must contain a JSON array of query objects.")

        queries: List[BenchmarkQuery] = []
        category_counts: Dict[str, int] = {}

        for item in raw_data:
            q = BenchmarkQuery(
                id=str(item.get("id", f"q_{len(queries)+1}")),
                query=str(item.get("query", "")).strip(),
                language=str(item.get("language", "en")).strip().lower(),
                category=str(item.get("category", "in_domain")).strip().lower(),
                expected_status=str(item.get("expected_status", "SUCCESS")).strip(),
                keywords=item.get("keywords", []),
            )
            queries.append(q)
            category_counts[q.category] = category_counts.get(q.category, 0) + 1

        # Validate total count
        if len(queries) != EXPECTED_TOTAL_QUERIES:
            logger.warning(
                "Benchmark query count is %d (Expected: %d). Proceeding with loaded queries.",
                len(queries),
                EXPECTED_TOTAL_QUERIES,
            )

        # Validate 5 category counts
        logger.info("Loaded %d benchmark queries with category counts: %s", len(queries), category_counts)
        for cat, expected_cnt in EXPECTED_CATEGORY_COUNTS.items():
            actual_cnt = category_counts.get(cat, 0)
            if actual_cnt != expected_cnt:
                logger.warning("Category '%s' count is %d (Expected: %d)", cat, actual_cnt, expected_cnt)

        return queries

    def run_warmup(self, num_warmup: int = 5) -> None:
        """Run warm-up queries to initialize ONNX runtime and database caches."""
        logger.info("Executing %d warm-up queries to prime ONNX engine and caches...", num_warmup)
        warmup_candidates = self.queries[:num_warmup] if self.queries else [
            BenchmarkQuery(id="w1", query="Warmup test query", language="en", category="in_domain")
        ]

        for q in warmup_candidates:
            try:
                _ = self.indexer.search(q.query, top_k=self.config.BENCHMARK_TOP_K)
            except Exception as e:
                logger.debug("Warm-up exception ignored: %s", e)

        logger.info("Warm-up phase completed successfully.")

    def run_query(self, query_obj: BenchmarkQuery) -> QueryExecutionResult:
        """Execute a single query with sub-stage latency measurement."""
        query_text = query_obj.query

        t_total_start = time.perf_counter()
        t_embed_start = time.perf_counter()

        try:
            # 1. Measure Embedding Latency
            query_vector = list(self.indexer.embedder.embed([query_text]))[0]
            embed_latency_ms = round((time.perf_counter() - t_embed_start) * 1000.0, 2)

            # 2. Measure Vector Search Latency
            t_search_start = time.perf_counter()
            hits = []
            if hasattr(self.indexer.client, "query_points"):
                resp = self.indexer.client.query_points(
                    collection_name=self.indexer.collection_name,
                    query=list(query_vector),
                    limit=self.config.BENCHMARK_TOP_K,
                    with_payload=True,
                )
                hits = resp.points
            elif hasattr(self.indexer.client, "search"):
                hits = self.indexer.client.search(
                    collection_name=self.indexer.collection_name,
                    query_vector=list(query_vector),
                    limit=self.config.BENCHMARK_TOP_K,
                    with_payload=True,
                )
            search_latency_ms = round((time.perf_counter() - t_search_start) * 1000.0, 2)
            total_latency_ms = round((time.perf_counter() - t_total_start) * 1000.0, 2)

            top_score = round(float(hits[0].score), 4) if hits else 0.0

            return QueryExecutionResult(
                query_id=query_obj.id,
                query=query_text,
                language=query_obj.language,
                category=query_obj.category,
                expected_status=query_obj.expected_status,
                success=True,
                embedding_latency_ms=embed_latency_ms,
                search_latency_ms=search_latency_ms,
                total_latency_ms=total_latency_ms,
                hits_count=len(hits),
                top_score=top_score,
            )

        except Exception as err:
            total_latency_ms = round((time.perf_counter() - t_total_start) * 1000.0, 2)
            logger.error("Query '%s' execution failed: %s", query_obj.id, err)
            return QueryExecutionResult(
                query_id=query_obj.id,
                query=query_text,
                language=query_obj.language,
                category=query_obj.category,
                expected_status=query_obj.expected_status,
                success=False,
                embedding_latency_ms=0.0,
                search_latency_ms=0.0,
                total_latency_ms=total_latency_ms,
                hits_count=0,
                top_score=0.0,
                error_message=str(err),
            )

    def run_benchmark(self, warmup: bool = True) -> Tuple[BenchmarkReport, List[QueryExecutionResult]]:
        """Run the complete 100-query benchmark suite.

        Returns tuple of (BenchmarkReport, List[QueryExecutionResult]).
        """
        logger.info("=" * 70)
        logger.info("  STARTING 100-QUERY MULTILINGUAL BENCHMARK SUITE")
        logger.info("  Model: %s | Dim: %d | Metric: %s", self.indexer.model_name, self.indexer.dimension, self.config.DISTANCE_METRIC)
        logger.info("  Collection: %s | Total Queries: %d", self.indexer.collection_name, len(self.queries))
        logger.info("=" * 70)

        if warmup:
            self.run_warmup(self.config.BENCHMARK_WARMUP_RUNS)

        results: List[QueryExecutionResult] = []
        category_latencies: Dict[str, List[float]] = {}
        language_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}

        total_latencies: List[float] = []
        embed_latencies: List[float] = []
        search_latencies: List[float] = []

        successful_queries = 0
        failed_queries = 0

        for idx, q in enumerate(self.queries, start=1):
            res = self.run_query(q)
            results.append(res)

            category_counts[q.category] = category_counts.get(q.category, 0) + 1
            language_counts[q.language] = language_counts.get(q.language, 0) + 1

            if res.success:
                successful_queries += 1
                total_latencies.append(res.total_latency_ms)
                embed_latencies.append(res.embedding_latency_ms)
                search_latencies.append(res.search_latency_ms)

                if q.category not in category_latencies:
                    category_latencies[q.category] = []
                category_latencies[q.category].append(res.total_latency_ms)
            else:
                failed_queries += 1

            if idx % 20 == 0 or idx == len(self.queries):
                logger.info("Executed %d/%d queries (Success: %d, Failed: %d)", idx, len(self.queries), successful_queries, failed_queries)

        # Statistical Calculations
        total_percentiles = calculate_percentiles(total_latencies)
        embed_percentiles = calculate_percentiles(embed_latencies)
        search_percentiles = calculate_percentiles(search_latencies)

        # Category Breakdown
        cat_breakdown: Dict[str, Dict[str, Any]] = {}
        for cat, lat_list in category_latencies.items():
            cat_pct = calculate_percentiles(lat_list)
            cat_breakdown[cat] = {
                "count": len(lat_list),
                "p50_ms": cat_pct.p50_ms,
                "p70_ms": cat_pct.p70_ms,
                "p100_ms": cat_pct.p100_ms,
                "mean_ms": cat_pct.mean_ms,
                "min_ms": cat_pct.min_ms,
                "max_ms": cat_pct.max_ms,
            }

        success_rate = round((successful_queries / max(1, len(self.queries))) * 100.0, 2)

        report = BenchmarkReport(
            total_queries=len(self.queries),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            success_rate_percent=success_rate,
            category_counts=category_counts,
            language_counts=language_counts,
            total_latency=total_percentiles,
            embedding_latency=embed_percentiles,
            search_latency=search_percentiles,
            category_breakdown=cat_breakdown,
            model_name=self.indexer.model_name,
            embedding_dimension=self.indexer.dimension,
            distance_metric=self.config.DISTANCE_METRIC,
            collection_name=self.indexer.collection_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        )

        return report, results

    def export_results(
        self,
        report: BenchmarkReport,
        results: List[QueryExecutionResult],
        json_path: Optional[Path] = None,
        csv_path: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """Export machine-readable JSON and tabular CSV reports."""
        target_json = json_path or (self.config.RESULTS_DIR / "benchmark_results.json")
        target_csv = csv_path or (self.config.RESULTS_DIR / "benchmark_summary.csv")

        target_json.parent.mkdir(parents=True, exist_ok=True)
        target_csv.parent.mkdir(parents=True, exist_ok=True)

        # 1. Export JSON Report
        full_json_payload = {
            "summary": report.to_dict(),
            "query_results": [r.to_dict() for r in results],
        }
        with open(target_json, "w", encoding="utf-8") as f:
            json.dump(full_json_payload, f, indent=2, ensure_ascii=False)
        logger.info("Exported JSON benchmark report to: %s", target_json)

        # 2. Export CSV Summary
        with open(target_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Count", "P50 (ms)", "P70 (ms)", "P100 (ms)", "Min (ms)", "Max (ms)", "Mean (ms)", "StdDev (ms)"])
            writer.writerow([
                "Total Retrieval Latency",
                report.total_latency.count,
                report.total_latency.p50_ms,
                report.total_latency.p70_ms,
                report.total_latency.p100_ms,
                report.total_latency.min_ms,
                report.total_latency.max_ms,
                report.total_latency.mean_ms,
                report.total_latency.stddev_ms,
            ])
            writer.writerow([
                "Query Embedding Latency",
                report.embedding_latency.count,
                report.embedding_latency.p50_ms,
                report.embedding_latency.p70_ms,
                report.embedding_latency.p100_ms,
                report.embedding_latency.min_ms,
                report.embedding_latency.max_ms,
                report.embedding_latency.mean_ms,
                report.embedding_latency.stddev_ms,
            ])
            writer.writerow([
                "Vector Search Latency",
                report.search_latency.count,
                report.search_latency.p50_ms,
                report.search_latency.p70_ms,
                report.search_latency.p100_ms,
                report.search_latency.min_ms,
                report.search_latency.max_ms,
                report.search_latency.mean_ms,
                report.search_latency.stddev_ms,
            ])

            writer.writerow([])
            writer.writerow(["Category", "Count", "P50 (ms)", "P70 (ms)", "P100 (ms)", "Min (ms)", "Max (ms)", "Mean (ms)", ""])
            for cat_name, stats in report.category_breakdown.items():
                writer.writerow([
                    cat_name,
                    stats.get("count", 0),
                    stats.get("p50_ms", 0.0),
                    stats.get("p70_ms", 0.0),
                    stats.get("p100_ms", 0.0),
                    stats.get("min_ms", 0.0),
                    stats.get("max_ms", 0.0),
                    stats.get("mean_ms", 0.0),
                    "",
                ])

        logger.info("Exported CSV benchmark summary to: %s", target_csv)
        return target_json, target_csv

    def print_console_summary(self, report: BenchmarkReport) -> None:
        """Print clean human-readable console report."""
        print("\n" + "=" * 78)
        print("  [BENCHMARK REPORT] 100-QUERY MULTILINGUAL BENCHMARK (DATA ANALYST)")
        print("=" * 78)
        print(f"  Model:        {report.model_name} ({report.embedding_dimension}-dim, {report.distance_metric})")
        print(f"  Collection:   {report.collection_name}")
        print(f"  Total:        {report.total_queries} queries (Success: {report.successful_queries}, Failed: {report.failed_queries}, Rate: {report.success_rate_percent}%)")
        print("-" * 78)
        print(f"  {'Stage / Metric':<28} | {'Count':<5} | {'P50 (ms)':<9} | {'P70 (ms)':<9} | {'P100 (ms)':<9} | {'Mean (ms)':<9}")
        print("-" * 78)
        print(f"  {'Total Retrieval Latency':<28} | {report.total_latency.count:<5} | {report.total_latency.p50_ms:<9} | {report.total_latency.p70_ms:<9} | {report.total_latency.p100_ms:<9} | {report.total_latency.mean_ms:<9}")
        print(f"  {'  - Query Embedding':<28} | {report.embedding_latency.count:<5} | {report.embedding_latency.p50_ms:<9} | {report.embedding_latency.p70_ms:<9} | {report.embedding_latency.p100_ms:<9} | {report.embedding_latency.mean_ms:<9}")
        print(f"  {'  - Qdrant Vector Search':<28} | {report.search_latency.count:<5} | {report.search_latency.p50_ms:<9} | {report.search_latency.p70_ms:<9} | {report.search_latency.p100_ms:<9} | {report.search_latency.mean_ms:<9}")
        print("-" * 78)
        print("  CATEGORY LATENCY BREAKDOWN (5 Categories):")
        for cat, stats in report.category_breakdown.items():
            print(f"    * {cat:<18} ({stats['count']:>2} queries) -> P50: {stats['p50_ms']:>6.2f} ms | P70: {stats['p70_ms']:>6.2f} ms | P100: {stats['p100_ms']:>6.2f} ms")
        print("=" * 78 + "\n")


# =============================================================================
# CLI Interface
# =============================================================================
def main() -> None:
    """CLI entry point for running the 100-query benchmark suite."""
    parser = argparse.ArgumentParser(description="Run 100-query benchmark on Qdrant vector index.")
    parser.add_argument("--mode", type=str, default="local", choices=["memory", "local", "remote"], help="Qdrant connection mode")
    parser.add_argument("--collection", type=str, default=None, help="Target Qdrant collection name")
    parser.add_argument("--queries-file", type=str, default=None, help="Path to 100-query JSON file")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warm-up iterations")
    args = parser.parse_args()

    config = get_pipeline_config()
    if args.mode:
        config.QDRANT_MODE = args.mode
    if args.collection:
        config.QDRANT_COLLECTION_NAME = args.collection

    indexer = VectorIndexer(config=config)
    suite = BenchmarkSuite(
        indexer=indexer,
        config=config,
        queries_file=Path(args.queries_file) if args.queries_file else None,
    )
    report, results = suite.run_benchmark(warmup=not args.no_warmup)
    suite.print_console_summary(report)
    json_path, csv_path = suite.export_results(report, results)
    print(f"[SUCCESS] Benchmark reports exported to:\n  - {json_path}\n  - {csv_path}")


if __name__ == "__main__":
    main()
