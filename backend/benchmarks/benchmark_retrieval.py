"""Benchmark script comparing multiple chunking strategies on MSMARCO-XI data."""

import asyncio
import json
import os
import time
from typing import Any, Dict, List
import numpy as np

from ingestion.chunker import get_chunker
from ingestion.dataset_loader import DatasetLoader
from ingestion.embedder import BatchEmbedder
from ingestion.indexer import QdrantIndexer


async def benchmark_chunking_strategies(data_file: str) -> Dict[str, Any]:
    """Compare Fixed-Size, Semantic Sentence, and Metadata-Aware chunking."""
    loader = DatasetLoader()
    documents = loader.load_from_jsonl(data_file)

    strategies = [
        ("fixed", "Strategy A: Fixed Size (200w, 40 overlap)"),
        ("semantic", "Strategy B: Semantic Sentence (150w target)"),
        ("metadata", "Strategy C: Metadata-Aware Structure"),
    ]

    results: Dict[str, Any] = {}

    for strat_key, strat_label in strategies:
        chunker = get_chunker(strat_key)

        t0 = time.perf_counter()
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk(
                text=doc.text,
                document_id=doc.document_id,
                metadata={"title": doc.title, "language": doc.language, "source": doc.source},
            )
            all_chunks.extend(chunks)
        chunk_time_ms = (time.perf_counter() - t0) * 1000.0

        lengths = [c.word_count for c in all_chunks]

        results[strat_key] = {
            "strategy_label": strat_label,
            "documents_count": len(documents),
            "chunks_count": len(all_chunks),
            "avg_chunks_per_doc": round(len(all_chunks) / len(documents), 2) if documents else 0,
            "avg_chunk_words": round(float(np.mean(lengths)), 1) if lengths else 0,
            "min_chunk_words": int(np.min(lengths)) if lengths else 0,
            "max_chunk_words": int(np.max(lengths)) if lengths else 0,
            "std_chunk_words": round(float(np.std(lengths)), 1) if lengths else 0,
            "chunking_time_ms": round(chunk_time_ms, 2),
        }

    return results


async def main() -> None:
    data_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample_msmarco_xi.jsonl")
    res = await benchmark_chunking_strategies(data_file)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
