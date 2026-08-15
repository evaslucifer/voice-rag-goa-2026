"""Tests for latency tracking utilities."""

import time
from app.utils.latency import LatencyTracker, LatencyBreakdown


def test_latency_tracker_measurement() -> None:
    """Test timing a stage with context manager."""
    tracker = LatencyTracker()

    with tracker.measure("embedding"):
        time.sleep(0.01)  # Sleep ~10ms

    embedding_latency = tracker.get_stage_latency("embedding")
    assert embedding_latency >= 5.0  # Should be at least ~5-10ms
    assert tracker.get_stage_latency("stt") == 0.0  # Unmeasured stages default to 0


def test_latency_tracker_manual_record() -> None:
    """Test explicitly recording stage latency."""
    tracker = LatencyTracker()
    tracker.record("retrieval", 24.567)

    assert tracker.get_stage_latency("retrieval") == 24.57


def test_latency_breakdown_schema_export() -> None:
    """Test conversion to LatencyBreakdown model."""
    tracker = LatencyTracker()
    tracker.record("stt", 40.0)
    tracker.record("embedding", 12.0)
    tracker.record("retrieval", 18.0)
    tracker.record("guardrail", 5.0)
    tracker.record("llm_ttft", 90.0)

    breakdown = tracker.to_breakdown()
    assert isinstance(breakdown, LatencyBreakdown)
    assert breakdown.stt == 40.0
    assert breakdown.embedding == 12.0
    assert breakdown.retrieval == 18.0
    assert breakdown.guardrail == 5.0
    assert breakdown.llm_ttft == 90.0
    assert breakdown.total >= 0.0

    breakdown_dict = tracker.to_dict()
    assert isinstance(breakdown_dict, dict)
    assert "total" in breakdown_dict
