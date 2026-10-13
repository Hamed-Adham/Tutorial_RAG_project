"""
Unit Tests for OperationalMetricsTracker.
Feature: 004-concurrency-gateway-sessions (Task T024)
"""

import pytest
from app.core.telemetry import OperationalMetricsTracker


def test_metrics_tracker_request_lifecycle():
    tracker = OperationalMetricsTracker(concurrency_limit=50)

    # Initial snapshot
    snap0 = tracker.get_metrics_snapshot()
    assert snap0.total_requests == 0
    assert snap0.active_requests == 0
    assert snap0.available_semaphore_slots == 50

    # Start 3 requests
    t1 = tracker.record_request_start()
    t2 = tracker.record_request_start()
    t3 = tracker.record_request_start()

    snap_mid = tracker.get_metrics_snapshot()
    assert snap_mid.active_requests == 3
    assert snap_mid.total_requests == 3
    assert snap_mid.available_semaphore_slots == 47

    # Finish requests
    tracker.record_request_finish(t1, queue_wait_ms=10.0, total_latency_ms=100.0, circuit_breaker_tripped=False, cache_hit=False)
    tracker.record_request_finish(t2, queue_wait_ms=20.0, total_latency_ms=200.0, circuit_breaker_tripped=False, cache_hit=True)
    tracker.record_request_finish(t3, queue_wait_ms=30.0, total_latency_ms=300.0, circuit_breaker_tripped=True, cache_hit=False)

    snap_done = tracker.get_metrics_snapshot()
    assert snap_done.active_requests == 0
    assert snap_done.total_requests == 3
    assert snap_done.available_semaphore_slots == 50
    assert snap_done.avg_queue_wait_ms == 20.0  # (10 + 20 + 30) / 3
    assert snap_done.circuit_breaker_trips == 1
    assert round(snap_done.circuit_breaker_trip_rate, 2) == 0.33
    assert snap_done.cache_hits == 1
    assert round(snap_done.cache_hit_rate, 2) == 0.33


def test_latency_percentiles():
    tracker = OperationalMetricsTracker(concurrency_limit=50)
    # Record 100 sample latencies from 10ms to 1000ms
    for i in range(1, 101):
        t0 = tracker.record_request_start()
        tracker.record_request_finish(t0, queue_wait_ms=1.0, total_latency_ms=float(i * 10), circuit_breaker_tripped=False, cache_hit=False)

    snap = tracker.get_metrics_snapshot()
    assert snap.total_requests == 100
    assert 480.0 <= snap.latency_p50_ms <= 520.0  # Median ~500ms
    assert 940.0 <= snap.latency_p95_ms <= 960.0  # p95 ~950ms
    assert snap.latency_p99_ms >= 980.0           # p99 ~990ms


def test_est_cost_per_1k_usd():
    tracker = OperationalMetricsTracker(concurrency_limit=50)
    # Record 1 request with 1,000,000 prompt tokens ($0.15) and 0 completion tokens
    t0 = tracker.record_request_start()
    tracker.record_request_finish(t0, queue_wait_ms=0.0, total_latency_ms=500.0, circuit_breaker_tripped=False, cache_hit=False, prompt_tokens=1_000_000, completion_tokens=0)

    snap = tracker.get_metrics_snapshot()
    # 1 request cost $0.15 -> cost per 1k requests = $0.15 * 1000 = $150.0
    assert snap.est_cost_per_1k_usd == 150.0
