"""Comprehensive Latency Benchmark Suite measuring all pipeline stages across 100+ queries."""

import asyncio
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

from app.schemas.response import QueryResponse
from app.services.cache_service import CacheService
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.guardrail_service import GuardrailService
from app.services.llm_service import LLMService
from app.services.qdrant_service import QdrantService, get_qdrant_service
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.utils.latency import LatencyTracker
from app.utils.logging import configure_logging, get_logger
from ingestion.chunker import get_chunker
from ingestion.embedder import BatchEmbedder
from ingestion.indexer import QdrantIndexer
from ingestion.data_provider import DataProvider, get_data_provider

logger = get_logger("benchmark.latency")


async def ensure_active_data_indexed(data_provider: Optional[DataProvider] = None) -> None:
    """Ensure current active dataset (demo or production) is indexed in Qdrant before benchmarking."""
    provider = data_provider or get_data_provider()
    records = provider.load_documents()
    if not records:
        logger.warning("No records found to index for data mode: %s", provider.data_mode)
        return

    collection_name = provider.get_target_collection()
    chunker = get_chunker("semantic")
    chunks = []
    for r in records:
        meta = {
            "language": r.language,
            "title": r.title,
            "query_id": r.metadata.get("query_id", r.document_id),
            "passage_id": r.document_id,
            "category": r.metadata.get("category", "in_domain"),
            "data_mode": r.metadata.get("data_mode", "DEMO DATA — NOT FINAL MSMARCO-XI DATA"),
        }
        chunks.extend(chunker.chunk(r.text, document_id=r.document_id, metadata=meta))

    embedder = BatchEmbedder()
    chunk_vector_pairs = await embedder.embed_chunks(chunks)
    indexer = QdrantIndexer(qdrant_service=get_qdrant_service(), collection_name=collection_name)
    await indexer.initialize_collection(recreate=False)
    await indexer.index_chunks(chunk_vector_pairs)


async def run_latency_benchmark(
    queries_file: Optional[str] = None,
    results_dir: str = "results",
    data_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute benchmark measuring raw per-stage latency for Uncached and Cached flows."""
    configure_logging(log_level="INFO", json_format=False)
    provider = get_data_provider(data_mode=data_mode)
    base_dir = os.path.dirname(__file__)

    if queries_file:
        queries_path = os.path.join(base_dir, queries_file) if not os.path.isabs(queries_file) else queries_file
    else:
        queries_path = provider.get_queries_path()

    output_dir = os.path.join(base_dir, results_dir) if not os.path.isabs(results_dir) else results_dir
    os.makedirs(output_dir, exist_ok=True)

    await ensure_active_data_indexed(data_provider=provider)

    with open(queries_path, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    mode_label = "DEMO RESULT — NOT FINAL MSMARCO-XI BENCHMARK" if provider.is_demo else "PRODUCTION MSMARCO-XI BENCHMARK"
    logger.info("[%s] Loaded %d benchmark queries from %s", mode_label, len(test_queries), queries_path)

    cache = CacheService()
    rag_service = RAGService(cache_service=cache)

    uncached_records: List[Dict[str, Any]] = []
    cached_records: List[Dict[str, Any]] = []

    # =========================================================================
    # Phase 1: Uncached Benchmark (Strict Cache Clear per query)
    # =========================================================================
    logger.info("--- Starting Phase 1: Uncached Full Pipeline Benchmark ---")
    for idx, item in enumerate(test_queries):
        cache.clear()
        q_id = item.get("id", f"q_{idx}")
        query_text = item["query"]
        lang = item.get("language", "en")
        category = item.get("category", "in_domain")

        tracker = LatencyTracker()
        req_id = f"bench-uncached-{q_id}"

        res: QueryResponse = await rag_service.execute_rag(
            query=query_text,
            request_id=req_id,
            language=lang,
            tracker=tracker,
        )

        uncached_records.append({
            "request_id": req_id,
            "query_id": q_id,
            "query": query_text,
            "language": lang,
            "query_type": category,
            "cache_hit": False,
            "status": res.status,
            "confidence_score": float(res.confidence_score),
            "citations_count": len(res.citations),
            "stt_ms": float(tracker.get_stage_latency("stt")),
            "guardrail_ms": float(tracker.get_stage_latency("guardrail")),
            "embedding_ms": float(tracker.get_stage_latency("embedding")),
            "retrieval_ms": float(tracker.get_stage_latency("retrieval")),
            "llm_ttft_ms": float(tracker.get_stage_latency("llm_ttft")),
            "total_ms": float(tracker.get_total_latency()),
        })

    # =========================================================================
    # Phase 2: Cached Benchmark (Query repeated with warm in-memory cache)
    # =========================================================================
    logger.info("--- Starting Phase 2: Cached Warm Pipeline Benchmark ---")
    # First ensure cache is primed for all test queries
    for item in test_queries:
        await rag_service.execute_rag(
            query=item["query"],
            request_id=f"warm-{item.get('id', 'q')}",
            language=item.get("language", "en"),
        )

    # Now measure the pure cached retrieval speed
    for idx, item in enumerate(test_queries):
        q_id = item.get("id", f"q_{idx}")
        query_text = item["query"]
        lang = item.get("language", "en")
        category = item.get("category", "in_domain")

        tracker = LatencyTracker()
        req_id = f"bench-cached-{q_id}"

        res: QueryResponse = await rag_service.execute_rag(
            query=query_text,
            request_id=req_id,
            language=lang,
            tracker=tracker,
        )

        cached_records.append({
            "request_id": req_id,
            "query_id": q_id,
            "query": query_text,
            "language": lang,
            "query_type": category,
            "cache_hit": True,
            "status": res.status,
            "confidence_score": float(res.confidence_score),
            "citations_count": len(res.citations),
            "stt_ms": float(tracker.get_stage_latency("stt")),
            "guardrail_ms": float(tracker.get_stage_latency("guardrail")),
            "embedding_ms": float(tracker.get_stage_latency("embedding")),
            "retrieval_ms": float(tracker.get_stage_latency("retrieval")),
            "llm_ttft_ms": float(tracker.get_stage_latency("llm_ttft")),
            "total_ms": float(tracker.get_total_latency()),
        })

    # =========================================================================
    # Phase 3: Percentile Analytics
    # =========================================================================
    def calc_percentiles(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        stages = ["embedding_ms", "retrieval_ms", "guardrail_ms", "llm_ttft_ms", "total_ms"]
        stats: Dict[str, Any] = {}
        for s in stages:
            vals = [r[s] for r in records]
            stats[s] = {
                "p50": float(np.percentile(vals, 50)),
                "p70": float(np.percentile(vals, 70)),
                "p90": float(np.percentile(vals, 90)),
                "p100": float(np.max(vals)),
                "avg": float(np.mean(vals)),
                "min": float(np.min(vals)),
            }
        return stats

    uncached_stats = calc_percentiles(uncached_records)
    cached_stats = calc_percentiles(cached_records)

    raw_results = {
        "metadata": {
            "data_mode": provider.data_mode,
            "benchmark_type": "DEMO RESULT — NOT FINAL MSMARCO-XI BENCHMARK" if provider.is_demo else "PRODUCTION MSMARCO-XI BENCHMARK",
            "total_queries": len(test_queries),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "hardware": "CPU Local Execution",
            "model_embedding": "BAAI/bge-small-en-v1.5 ONNX",
            "retrieval_db": f"Qdrant ({provider.get_target_collection()})",
            "target_p50_ms": 200.0,
        },
        "summary": {
            "uncached": uncached_stats,
            "cached": cached_stats,
        },
        "raw_uncached_records": uncached_records,
        "raw_cached_records": cached_records,
    }

    # Write raw unrounded JSON
    json_path = os.path.join(output_dir, "latency_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)
    logger.info("Saved raw latency results to %s", json_path)

    # Write summary CSV
    csv_path = os.path.join(output_dir, "latency_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Benchmark_Mode", "Stage", "P50_ms", "P70_ms", "P90_ms", "P100_Max_ms", "Avg_ms", "Min_ms"])
        for stage_name, metrics in uncached_stats.items():
            writer.writerow([
                "Uncached",
                stage_name.replace("_ms", ""),
                f"{metrics['p50']:.3f}",
                f"{metrics['p70']:.3f}",
                f"{metrics['p90']:.3f}",
                f"{metrics['p100']:.3f}",
                f"{metrics['avg']:.3f}",
                f"{metrics['min']:.3f}",
            ])
        for stage_name, metrics in cached_stats.items():
            writer.writerow([
                "Cached",
                stage_name.replace("_ms", ""),
                f"{metrics['p50']:.3f}",
                f"{metrics['p70']:.3f}",
                f"{metrics['p90']:.3f}",
                f"{metrics['p100']:.3f}",
                f"{metrics['avg']:.3f}",
                f"{metrics['min']:.3f}",
            ])
    logger.info("Saved latency summary CSV to %s", csv_path)

    logger.info("=================================================================")
    logger.info("BENCHMARK COMPLETED (%d queries)", len(test_queries))
    logger.info("Uncached E2E Latency: P50 = %.2f ms | P70 = %.2f ms | P100 = %.2f ms", uncached_stats["total_ms"]["p50"], uncached_stats["total_ms"]["p70"], uncached_stats["total_ms"]["p100"])
    logger.info("Cached E2E Latency:   P50 = %.2f ms | P70 = %.2f ms | P100 = %.2f ms", cached_stats["total_ms"]["p50"], cached_stats["total_ms"]["p70"], cached_stats["total_ms"]["p100"])
    logger.info("=================================================================")

    return raw_results


if __name__ == "__main__":
    asyncio.run(run_latency_benchmark())
