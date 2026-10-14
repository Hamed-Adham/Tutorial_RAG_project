"""
Operational Telemetry & Production Health Metrics Tracker.
Tracks active workers, queue wait ms, circuit breaker trip rates, cache hits, and latency percentiles.
"""

import time
import bisect
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from app.contracts.gateway_contracts import OperationalMetricsModel, IOperationalMetricsTracker
from app.core.config import settings


class OperationalMetricsTracker(IOperationalMetricsTracker):
    """
    Thread-safe, sub-millisecond in-memory metrics aggregator.
    Maintains running counters, queue delay averages, and latency histograms.
    """

    def __init__(self, concurrency_limit: int = settings.concurrency_limit):
        self._lock = threading.Lock()
        self.concurrency_limit = concurrency_limit
        self._active_requests = 0
        self._total_requests = 0
        self._circuit_breaker_trips = 0
        self._cache_hits = 0
        self._total_queue_wait_ms = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._latencies: List[float] = []  # Sorted list for percentile calculations
        self._start_time = time.time()

    def record_request_start(self) -> float:
        """Mark start timestamp and increment active requests counter."""
        with self._lock:
            self._active_requests += 1
            self._total_requests += 1
        return time.perf_counter()

    def record_request_finish(
        self,
        start_time: float,
        queue_wait_ms: float,
        total_latency_ms: float,
        circuit_breaker_tripped: bool,
        cache_hit: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ) -> None:
        """Record completed query telemetry and update running latency percentiles."""
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._total_queue_wait_ms += queue_wait_ms
            if circuit_breaker_tripped:
                self._circuit_breaker_trips += 1
            if cache_hit:
                self._cache_hits += 1
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens

            # Insert into sorted list for O(log N) percentile lookup (capped at last 1000)
            bisect.insort(self._latencies, total_latency_ms)
            if len(self._latencies) > 1000:
                self._latencies.pop(0)

    @staticmethod
    def calculate_cost_1k(prompt_tokens: int, completion_tokens: int, cache_hit: bool = False) -> Dict[str, Any]:
        """
        Calculate single query cost and projected Cost / 1k Q for:
        - Min Model Tier (Flash/Mini):
          • Input Miss: $0.15 per 1M tokens
          • Input Hit:  $0.003 per 1M tokens
          • Output:     $0.60 per 1M tokens
        - Max Model Tier (Flagship/CoT):
          • Input Miss: $1.50 per 1M tokens
          • Input Hit:  $0.00 per 1M tokens (free caching)
          • Output:     $7.50 per 1M tokens
        """
        p_miss = 0 if cache_hit else prompt_tokens
        p_hit = prompt_tokens if cache_hit else 0
        min_cost = (p_miss * 0.15 / 1_000_000.0) + (p_hit * 0.003 / 1_000_000.0) + (completion_tokens * 0.60 / 1_000_000.0)
        max_cost = (p_miss * 1.50 / 1_000_000.0) + (p_hit * 0.00 / 1_000_000.0) + (completion_tokens * 7.50 / 1_000_000.0)
        return {
            "min_1k_usd": round(min_cost * 1000.0, 3),
            "max_1k_usd": round(max_cost * 1000.0, 3),
            "cost_single_min_usd": round(min_cost, 6),
            "cost_single_max_usd": round(max_cost, 6),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def _get_percentile(self, p: float) -> float:
        if not self._latencies:
            return 0.0
        k = (len(self._latencies) - 1) * p
        f = int(k)
        c = min(f + 1, len(self._latencies) - 1)
        d = k - f
        return round(self._latencies[f] + d * (self._latencies[c] - self._latencies[f]), 2)

    def get_metrics_snapshot(self) -> OperationalMetricsModel:
        """Return point-in-time metrics snapshot for /api/v1/metrics."""
        with self._lock:
            total = max(1, self._total_requests)
            cbr = self._circuit_breaker_trips / total
            chr_rate = self._cache_hits / total
            avg_queue = self._total_queue_wait_ms / total

            # Split tokens by cache hit rate
            p_hit = self._total_prompt_tokens * chr_rate
            p_miss = self._total_prompt_tokens * (1.0 - chr_rate)
            c_tok = self._total_completion_tokens

            # Min Model Tier ($0.15 miss, $0.003 hit, $0.60 out per 1M)
            min_input_cost = (p_miss / 1_000_000.0) * 0.15 + (p_hit / 1_000_000.0) * 0.003
            min_output_cost = (c_tok / 1_000_000.0) * 0.60
            min_cost_per_req = (min_input_cost + min_output_cost) / total
            est_cost_min_1k = round(min_cost_per_req * 1000.0, 3)

            # Max Model Tier ($1.50 miss, free caching $0.00 hit, $7.50 out per 1M)
            max_input_cost = (p_miss / 1_000_000.0) * 1.50
            max_output_cost = (c_tok / 1_000_000.0) * 7.50
            max_cost_per_req = (max_input_cost + max_output_cost) / total
            est_cost_max_1k = round(max_cost_per_req * 1000.0, 3)

            return OperationalMetricsModel(
                active_requests=self._active_requests,
                available_semaphore_slots=max(0, self.concurrency_limit - self._active_requests),
                avg_queue_wait_ms=round(avg_queue, 2),
                total_requests=self._total_requests,
                circuit_breaker_trips=self._circuit_breaker_trips,
                circuit_breaker_trip_rate=round(cbr, 4),
                cache_hits=self._cache_hits,
                cache_hit_rate=round(chr_rate, 4),
                latency_p50_ms=self._get_percentile(0.50),
                latency_p95_ms=self._get_percentile(0.95),
                latency_p99_ms=self._get_percentile(0.99),
                est_cost_per_1k_usd=est_cost_min_1k,
                est_cost_per_1k_min_usd=est_cost_min_1k,
                est_cost_per_1k_max_usd=est_cost_max_1k,
            )


# Global singleton tracker
metrics_tracker = OperationalMetricsTracker()
