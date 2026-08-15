"""Retrieval & Chunking Strategy Benchmark comparing Semantic, Parent-Child, Metadata, and Fixed chunking."""

import asyncio
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.qdrant_service import QdrantService, get_qdrant_service
from app.services.retrieval_service import RetrievalService, RetrievedChunk
from app.utils.logging import configure_logging, get_logger
from ingestion.chunker import get_chunker
from ingestion.data_provider import DataProvider, get_data_provider
from ingestion.embedder import BatchEmbedder
from ingestion.indexer import QdrantIndexer

logger = get_logger("benchmark.retrieval")


async def benchmark_chunking_strategies(
    queries_file: Optional[str] = None,
    results_dir: str = "results",
    data_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Index active corpus under each strategy and benchmark retrieval metrics."""
    configure_logging(log_level="INFO", json_format=False)
    provider = get_data_provider(data_mode=data_mode)
    base_dir = os.path.dirname(__file__)

    if queries_file:
        queries_path = os.path.join(base_dir, queries_file) if not os.path.isabs(queries_file) else queries_file
    else:
        queries_path = provider.get_queries_path()

    output_dir = os.path.join(base_dir, results_dir) if not os.path.isabs(results_dir) else results_dir
    os.makedirs(output_dir, exist_ok=True)

    records = provider.load_documents()

    with open(queries_path, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    # Use in-domain queries for retrieval quality evaluation
    eval_queries = [q for q in all_queries if q.get("category") == "in_domain"]
    if not eval_queries:
        eval_queries = all_queries[:30]

    strategies = ["semantic", "parent_child", "metadata", "fixed"]
    strategy_results: Dict[str, Any] = {}

    embedding_service = get_embedding_service()
    qdrant_service = get_qdrant_service()
    batch_embedder = BatchEmbedder(embedding_service=embedding_service)

    for strat in strategies:
        prefix = "demo_strat" if provider.is_demo else "bench_strat"
        col_name = f"{prefix}_{strat}"
        logger.info("Evaluating strategy: %s (collection=%s)...", strat, col_name)

        chunker = get_chunker(strat)
        chunks = []
        for r in records:
            meta = {
                "language": r.language,
                "title": r.title,
                "query_id": r.metadata.get("query_id", r.document_id),
                "passage_id": r.document_id,
                "category": r.metadata.get("category", "in_domain"),
                "is_selected": r.metadata.get("is_selected", True),
                "source": r.source,
                "data_mode": r.metadata.get("data_mode", "DEMO DATA — NOT FINAL MSMARCO-XI DATA"),
                "chunk_strategy": strat,
            }
            chunks.extend(chunker.chunk(r.text, document_id=r.document_id, metadata=meta))

        # Embed and index
        chunk_vector_pairs = await batch_embedder.embed_chunks(chunks)
        indexer = QdrantIndexer(qdrant_service=qdrant_service, collection_name=col_name)
        await indexer.initialize_collection(recreate=True)
        await indexer.index_chunks(chunk_vector_pairs)

        retrieval_service = RetrievalService(
            embedding_service=embedding_service,
            qdrant_service=qdrant_service,
            top_k=3,
            score_threshold=0.40,
        )

        query_embed_times: List[float] = []
        qdrant_search_times: List[float] = []
        top1_scores: List[float] = []
        top3_avg_scores: List[float] = []
        context_word_counts: List[int] = []

        for q in eval_queries:
            q_text = q["query"]
            res = await retrieval_service.retrieve(query=q_text, top_k=3, score_threshold=0.0, collection_name=col_name)

            query_embed_times.append(res.embedding_latency_ms)
            qdrant_search_times.append(res.retrieval_latency_ms)
            top1_scores.append(res.top_score)

            if res.chunks:
                top3_avg_scores.append(float(np.mean([c.score for c in res.chunks])))
            else:
                top3_avg_scores.append(0.0)

            context_word_counts.append(len(res.context_text.split()))

        strat_summary = {
            "strategy": strat,
            "chunk_count": len(chunks),
            "embed_p50_ms": float(np.percentile(query_embed_times, 50)),
            "search_p50_ms": float(np.percentile(qdrant_search_times, 50)),
            "retrieval_total_p50_ms": float(np.percentile(np.array(query_embed_times) + np.array(qdrant_search_times), 50)),
            "avg_top1_score": float(np.mean(top1_scores)),
            "avg_top3_score": float(np.mean(top3_avg_scores)),
            "avg_context_words": float(np.mean(context_word_counts)),
        }
        strategy_results[strat] = strat_summary
        logger.info(
            "Strategy %s: Retrieval P50 = %.2f ms | Search P50 = %.2f ms | Avg Top-1 Score = %.3f | Context Words = %.1f",
            strat,
            strat_summary["retrieval_total_p50_ms"],
            strat_summary["search_p50_ms"],
            strat_summary["avg_top1_score"],
            strat_summary["avg_context_words"],
        )

    # Save to CSV
    csv_path = os.path.join(output_dir, "chunking_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Strategy", "Chunk_Count", "Retrieval_P50_ms", "Search_P50_ms", "Avg_Top1_Score", "Avg_Top3_Score", "Avg_Context_Words"])
        for strat, data in strategy_results.items():
            writer.writerow([
                strat,
                data["chunk_count"],
                f"{data['retrieval_total_p50_ms']:.2f}",
                f"{data['search_p50_ms']:.2f}",
                f"{data['avg_top1_score']:.3f}",
                f"{data['avg_top3_score']:.3f}",
                f"{data['avg_context_words']:.1f}",
            ])
    logger.info("Saved chunking strategy comparison CSV to %s", csv_path)

    return strategy_results


if __name__ == "__main__":
    asyncio.run(benchmark_chunking_strategies())
