"""Dataset loader and record normalizer for ai4bharat/MSMARCO-XI."""

import json
import os
from typing import Any, Dict, Generator, Iterator, List, Optional
from pydantic import BaseModel, Field
from app.utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_INDIC_LANGUAGES = [
    "en", "hi", "te", "ta", "bn", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"
]


class NormalizedDocument(BaseModel):
    """Normalized document schema derived from MSMARCO-XI records."""

    document_id: str = Field(..., description="Unique document / passage identifier")
    text: str = Field(..., description="Passage or document body text")
    title: str = Field(default="", description="Document title if available")
    language: str = Field(default="en", description="Language code (e.g. en, hi, te, ta)")
    source: str = Field(default="ai4bharat/MSMARCO-XI", description="Dataset provenance")
    query: Optional[str] = Field(default=None, description="Paired query for benchmark evaluation")
    answers: List[str] = Field(default_factory=list, description="Ground-truth reference answers if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Raw metadata preserved from source record")


class DatasetLoader:
    """Loader for MSMARCO-XI and local benchmark datasets."""

    def __init__(self, default_language: str = "en") -> None:
        self.default_language = default_language

    def normalize_record(self, raw_record: Dict[str, Any], default_lang: str = "en") -> Optional[NormalizedDocument]:
        """Normalize a raw record from MSMARCO-XI or JSONL into NormalizedDocument.

        Returns None if record is invalid or empty.
        """
        if not isinstance(raw_record, dict):
            return None

        # Extract text from various possible schema variations
        text = (
            raw_record.get("passage")
            or raw_record.get("text")
            or raw_record.get("passage_text")
            or raw_record.get("body")
            or ""
        )
        if isinstance(text, list):
            text = " ".join([str(t) for t in text if t])
        text = str(text).strip()

        # Check for nested passages list format: [{"passage_id": ..., "text": ..., "is_selected": ...}]
        doc_id = None
        passages_list = raw_record.get("passages")
        if isinstance(passages_list, list) and passages_list and not text:
            first_passage = passages_list[0]
            if isinstance(first_passage, dict):
                text = str(first_passage.get("text") or first_passage.get("passage") or "").strip()
                p_id = first_passage.get("passage_id") or first_passage.get("id")
                if p_id:
                    doc_id = str(p_id)

        if not text:
            return None

        # Extract document ID if not already resolved from passage
        if not doc_id:
            doc_id = str(
                raw_record.get("passage_id")
                or raw_record.get("doc_id")
                or raw_record.get("id")
                or raw_record.get("document_id")
                or f"doc_{hash(text) & 0xFFFFFFFF}"
            )

        title = str(raw_record.get("title") or raw_record.get("heading") or "").strip()
        language = str(raw_record.get("language") or raw_record.get("lang") or default_lang).strip()

        query = raw_record.get("query") or raw_record.get("question")
        if query:
            query = str(query).strip()

        raw_answers = raw_record.get("answers") or raw_record.get("answer") or []
        if isinstance(raw_answers, str):
            answers = [raw_answers.strip()] if raw_answers.strip() else []
        elif isinstance(raw_answers, list):
            answers = [str(a).strip() for a in raw_answers if str(a).strip()]
        else:
            answers = []

        # Retain category, query_id, is_selected, and data_mode in metadata
        meta = {k: v for k, v in raw_record.items() if k not in ("passage", "text", "body", "passages")}
        if raw_record.get("id"):
            meta["query_id"] = str(raw_record["id"])
        if isinstance(passages_list, list) and passages_list and isinstance(passages_list[0], dict):
            meta["is_selected"] = passages_list[0].get("is_selected", True)
            meta["passage_id"] = doc_id

        return NormalizedDocument(
            document_id=doc_id,
            text=text,
            title=title,
            language=language,
            source=str(raw_record.get("source") or "ai4bharat/MSMARCO-XI"),
            query=query,
            answers=answers,
            metadata=meta,
        )

    def load_from_jsonl(self, file_path: str, max_records: Optional[int] = None) -> List[NormalizedDocument]:
        """Load and normalize documents from a local JSON Lines file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        documents: List[NormalizedDocument] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if max_records is not None and len(documents) >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    doc = self.normalize_record(data, default_lang=self.default_language)
                    if doc:
                        documents.append(doc)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON line at %s:%d", file_path, line_idx)

        logger.info("Loaded %d valid normalized documents from %s", len(documents), file_path)
        return documents

    def stream_hf_dataset(
        self,
        dataset_name: str = "ai4bharat/MSMARCO-XI",
        language_code: str = "en",
        split: str = "train",
        max_records: Optional[int] = None,
    ) -> Generator[NormalizedDocument, None, None]:
        """Stream normalized records directly from HuggingFace datasets."""
        try:
            from datasets import load_dataset
            logger.info("Connecting to Hugging Face dataset: %s (%s, split=%s)", dataset_name, language_code, split)
            ds = load_dataset(dataset_name, language_code, split=split, streaming=True)

            count = 0
            for raw_item in ds:
                if max_records is not None and count >= max_records:
                    break
                doc = self.normalize_record(dict(raw_item), default_lang=language_code)
                if doc:
                    count += 1
                    yield doc

            logger.info("Successfully streamed %d documents from %s (%s)", count, dataset_name, language_code)

        except Exception as e:
            logger.error("Failed to load HuggingFace dataset %s: %s", dataset_name, str(e), exc_info=True)
            raise


def load_from_jsonl(file_path: str, max_records: Optional[int] = None, default_lang: str = "en") -> List[NormalizedDocument]:
    """Helper function to load and normalize documents from JSONL."""
    loader = DatasetLoader(default_language=default_lang)
    return loader.load_from_jsonl(file_path=file_path, max_records=max_records)
