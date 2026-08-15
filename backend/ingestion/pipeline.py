"""End-to-end ingestion pipeline runner for MSMARCO-XI."""

import argparse
import asyncio
import time
from typing import Any, Dict, List, Optional
from app.config import get_settings
from app.services.embedding_service import get_embedding_service
from app.services.qdrant_service import get_qdrant_service
from app.utils.logging import configure_logging, get_logger
from ingestion.chunker import Chunk, get_chunker
from ingestion.dataset_loader import DatasetLoader, NormalizedDocument
from ingestion.embedder import BatchEmbedder
from ingestion.indexer import QdrantIndexer

logger = get_logger("ingestion.pipeline")


class IngestionPipeline:
    """Orchestrates dataset loading, chunking, embedding, and vector indexing."""

    def __init__(
        self,
        chunking_strategy: str = "semantic",
        collection_name: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.settings = get_settings()
        self.chunking_strategy = chunking_strategy
        self.chunker = get_chunker(strategy=chunking_strategy)
        self.dataset_loader = DatasetLoader()
        self.embedding_service = get_embedding_service()
        self.embedder = BatchEmbedder(embedding_service=self.embedding_service, batch_size=batch_size)
        self.qdrant_service = get_qdrant_service()
        self.indexer = QdrantIndexer(
            qdrant_service=self.qdrant_service,
            collection_name=collection_name or self.settings.QDRANT_COLLECTION,
            vector_dim=384,
        )

    async def run(
        self,
        documents: List[NormalizedDocument],
        recreate_collection: bool = False,
    ) -> Dict[str, Any]:
        """Execute the ingestion pipeline on a list of normalized documents."""
        pipeline_t0 = time.perf_counter()
        logger.info(
            "Starting ingestion for %d documents (strategy='%s', collection='%s')...",
            len(documents),
            self.chunking_strategy,
            self.indexer.collection_name,
        )

        # 1. Initialize collection in Qdrant
        await self.indexer.initialize_collection(recreate=recreate_collection)

        # 2. Chunk all documents
        chunk_t0 = time.perf_counter()
        all_chunks: List[Chunk] = []
        for doc in documents:
            doc_chunks = self.chunker.chunk(
                text=doc.text,
                document_id=doc.document_id,
                metadata={
                    "document_id": doc.document_id,
                    "title": doc.title,
                    "language": doc.language,
                    "source": doc.source,
                    "query": doc.query,
                    "answers": doc.answers,
                },
            )
            all_chunks.extend(doc_chunks)

        chunk_duration_ms = (time.perf_counter() - chunk_t0) * 1000.0
        logger.info("Generated %d chunks from %d documents in %.2f ms", len(all_chunks), len(documents), chunk_duration_ms)

        if not all_chunks:
            return {
                "status": "COMPLETED",
                "documents_count": len(documents),
                "chunks_count": 0,
                "indexed_count": 0,
                "total_time_ms": round((time.perf_counter() - pipeline_t0) * 1000.0, 2),
            }

        # 3. Generate embeddings
        embed_t0 = time.perf_counter()
        chunk_vector_pairs = await self.embedder.embed_chunks(all_chunks)
        embed_duration_ms = (time.perf_counter() - embed_t0) * 1000.0

        # 4. Index into Qdrant
        index_t0 = time.perf_counter()
        indexed_count = await self.indexer.index_chunks(chunk_vector_pairs)
        index_duration_ms = (time.perf_counter() - index_t0) * 1000.0

        total_duration_ms = (time.perf_counter() - pipeline_t0) * 1000.0

        summary = {
            "status": "COMPLETED",
            "strategy": self.chunking_strategy,
            "collection": self.indexer.collection_name,
            "documents_count": len(documents),
            "chunks_count": len(all_chunks),
            "indexed_count": indexed_count,
            "avg_chunks_per_doc": round(len(all_chunks) / len(documents), 2) if documents else 0,
            "timing_breakdown_ms": {
                "chunking": round(chunk_duration_ms, 2),
                "embedding": round(embed_duration_ms, 2),
                "indexing": round(index_duration_ms, 2),
                "total": round(total_duration_ms, 2),
            },
        }

        logger.info("Ingestion completed successfully: %s", summary)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest MSMARCO-XI dataset into Qdrant.")
    parser.add_argument("--strategy", type=str, default="semantic", choices=["semantic", "fixed", "metadata"], help="Chunking strategy")
    parser.add_argument("--file", type=str, default=None, help="Path to local JSONL dataset file")
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection")
    parser.add_argument("--max-records", type=int, default=100, help="Max records to ingest")
    parser.add_argument("--language", type=str, default="en", help="Language code (en, hi, te, etc.)")
    return parser.parse_args()


async def main() -> None:
    configure_logging(log_level="INFO", json_format=False)
    args = parse_args()
    pipeline = IngestionPipeline(chunking_strategy=args.strategy)

    loader = DatasetLoader(default_language=args.language)
    if args.file:
        documents = loader.load_from_jsonl(args.file, max_records=args.max_records)
    else:
        # Load from HuggingFace dataset
        documents = list(loader.stream_hf_dataset(language_code=args.language, max_records=args.max_records))

    await pipeline.run(documents=documents, recreate_collection=args.recreate)


if __name__ == "__main__":
    asyncio.run(main())
