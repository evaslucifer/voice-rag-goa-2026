"""Latency tracking and measurement utilities."""

import time
from contextlib import contextmanager
from typing import Dict, Generator, Optional
from pydantic import BaseModel, Field


class LatencyBreakdown(BaseModel):
    """Structured latency breakdown in milliseconds for each pipeline stage."""

    stt: float = Field(default=0.0, description="Sarvam STT latency in milliseconds")
    embedding: float = Field(default=0.0, description="FastEmbed query embedding latency in milliseconds")
    retrieval: float = Field(default=0.0, description="Qdrant vector retrieval latency in milliseconds")
    guardrail: float = Field(default=0.0, description="Guardrail checks latency in milliseconds")
    llm_ttft: float = Field(default=0.0, description="LLM Time to First Token (TTFT) in milliseconds")
    total: float = Field(default=0.0, description="Total end-to-end pipeline latency in milliseconds")


class LatencyTracker:
    """Utility for measuring real execution latency per stage in milliseconds."""

    def __init__(self) -> None:
        self._start_time: float = time.perf_counter()
        self._measurements: Dict[str, float] = {}

    @property
    def start_time(self) -> float:
        """Return the initialization start time."""
        return self._start_time

    @contextmanager
    def measure(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager to measure the elapsed time of a synchronous or async block."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._measurements[stage_name] = round(elapsed_ms, 2)

    def record(self, stage_name: str, duration_ms: float) -> None:
        """Explicitly record a measured duration in milliseconds."""
        self._measurements[stage_name] = round(duration_ms, 2)

    def get_stage_latency(self, stage_name: str) -> float:
        """Get the recorded latency for a specific stage, defaulting to 0.0."""
        return self._measurements.get(stage_name, 0.0)

    def get_total_latency(self) -> float:
        """Calculate total elapsed time since tracker initialization."""
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        return round(elapsed_ms, 2)

    def to_breakdown(self) -> LatencyBreakdown:
        """Generate a validated LatencyBreakdown model."""
        return LatencyBreakdown(
            stt=self.get_stage_latency("stt"),
            embedding=self.get_stage_latency("embedding"),
            retrieval=self.get_stage_latency("retrieval"),
            guardrail=self.get_stage_latency("guardrail"),
            llm_ttft=self.get_stage_latency("llm_ttft"),
            total=self.get_total_latency(),
        )

    def to_dict(self) -> Dict[str, float]:
        """Return the latency breakdown as a dictionary."""
        return self.to_breakdown().model_dump()
