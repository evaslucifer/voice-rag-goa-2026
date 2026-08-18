"""Unit tests for the 100-query benchmark suite and latency metrics."""

import json
from pathlib import Path
import pytest

from data_pipeline.benchmark_suite import (
    EXPECTED_CATEGORY_COUNTS,
    EXPECTED_TOTAL_QUERIES,
    BenchmarkQuery,
    BenchmarkSuite,
    calculate_percentiles,
)
from data_pipeline.vector_indexer import VectorIndexer


def test_calculate_percentiles_accuracy():
    """Test mathematical accuracy of P50, P70, P100, min, max, mean, and stddev."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    pct = calculate_percentiles(latencies)

    assert pct.count == 10
    assert pct.min_ms == 10.0
    assert pct.max_ms == 100.0
    assert pct.p50_ms == 50.0
    assert pct.p70_ms == 70.0
    assert pct.p100_ms == 100.0
    assert pct.mean_ms == 55.0
    assert pct.stddev_ms == 28.72


def test_calculate_percentiles_empty():
    """Test percentile computation with empty list returns zeros safely."""
    pct = calculate_percentiles([])
    assert pct.count == 0
    assert pct.p50_ms == 0.0
    assert pct.p70_ms == 0.0
    assert pct.p100_ms == 0.0


def test_benchmark_queries_file_category_counts(mock_config):
    """Test that benchmark_100_queries.json contains exactly 100 queries and exact 5 category counts."""
    queries_file = Path(__file__).resolve().parent.parent / "data" / "benchmark_100_queries.json"
    assert queries_file.exists(), f"Benchmark file missing at: {queries_file}"

    with open(queries_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == EXPECTED_TOTAL_QUERIES == 100

    category_counts = {}
    for item in data:
        cat = item.get("category")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    assert category_counts == EXPECTED_CATEGORY_COUNTS
    assert category_counts["in_domain"] == 50
    assert category_counts["ambiguous"] == 15
    assert category_counts["no_answer"] == 15
    assert category_counts["off_topic"] == 10
    assert category_counts["prompt_injection"] == 10


def test_run_benchmark_suite_in_memory(in_memory_indexer, sample_chunk_records, mock_config):
    """Test running 100-query benchmark suite on in-memory indexer and exporting reports."""
    # Index sample chunks first
    in_memory_indexer.index_chunks(sample_chunk_records, recreate_collection=True)

    suite = BenchmarkSuite(
        indexer=in_memory_indexer,
        config=mock_config,
    )

    report, results = suite.run_benchmark(warmup=False)

    assert report.total_queries == 100
    assert report.successful_queries == 100
    assert report.failed_queries == 0
    assert report.success_rate_percent == 100.0
    assert report.total_latency.p50_ms > 0.0
    assert report.total_latency.p70_ms >= report.total_latency.p50_ms
    assert report.total_latency.p100_ms >= report.total_latency.p70_ms

    # Test export
    json_path, csv_path = suite.export_results(report, results)
    assert json_path.exists()
    assert csv_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        exported_json = json.load(f)
        assert exported_json["summary"]["total_queries"] == 100
        assert len(exported_json["query_results"]) == 100
