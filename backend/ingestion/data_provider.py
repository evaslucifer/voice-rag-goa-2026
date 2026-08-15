"""Data provider abstraction for switching cleanly between Demo and Production datasets."""

import os
from typing import List, Optional
from app.config import get_settings
from app.utils.logging import get_logger
from ingestion.dataset_loader import DatasetLoader, NormalizedDocument

logger = get_logger(__name__)

# Base directory paths
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DATA_FILE = os.path.join(BACKEND_DIR, "data", "demo", "demo_msmarco_xi.jsonl")
PROD_SAMPLE_FILE = os.path.join(BACKEND_DIR, "data", "sample_msmarco_xi.jsonl")
DEMO_QUERIES_FILE = os.path.join(BACKEND_DIR, "benchmarks", "demo_test_queries.json")
PROD_QUERIES_FILE = os.path.join(BACKEND_DIR, "benchmarks", "test_queries.json")


class DataProvider:
    """Central data provider resolving dataset paths, query sets, and collections."""

    def __init__(self, data_mode: Optional[str] = None) -> None:
        self.settings = get_settings()
        self.data_mode = (data_mode or self.settings.DATA_MODE).lower()

    @property
    def is_demo(self) -> bool:
        """Check if currently configured for demo data."""
        return self.data_mode in ("demo", "test", "dev")

    def get_dataset_path(self) -> str:
        """Return the appropriate dataset JSONL file path based on data_mode."""
        if self.is_demo:
            if os.path.exists(DEMO_DATA_FILE):
                return DEMO_DATA_FILE
            logger.warning("Demo dataset file not found at %s; falling back to sample dataset", DEMO_DATA_FILE)
            return PROD_SAMPLE_FILE
        return PROD_SAMPLE_FILE

    def get_queries_path(self) -> str:
        """Return the appropriate benchmark queries JSON file path."""
        if self.is_demo:
            if os.path.exists(DEMO_QUERIES_FILE):
                return DEMO_QUERIES_FILE
            logger.warning("Demo queries file not found at %s; falling back to production query set", DEMO_QUERIES_FILE)
            return PROD_QUERIES_FILE
        return PROD_QUERIES_FILE

    def get_target_collection(self) -> str:
        """Return the target Qdrant collection name for current data mode."""
        if self.is_demo:
            return self.settings.QDRANT_COLLECTION or "msmarco_demo"
        return self.settings.QDRANT_COLLECTION or "msmarco_xi_bge_small"

    def load_documents(self, max_records: Optional[int] = None) -> List[NormalizedDocument]:
        """Load and normalize documents from the active dataset."""
        path = self.get_dataset_path()
        loader = DatasetLoader()
        logger.info("Loading documents for DATA_MODE='%s' from: %s", self.data_mode, path)
        return loader.load_from_jsonl(file_path=path, max_records=max_records)


def get_data_provider(data_mode: Optional[str] = None) -> DataProvider:
    """Return a DataProvider instance."""
    return DataProvider(data_mode=data_mode)
