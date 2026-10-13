"""
Unit & Concurrency Integration Tests for ConcurrencyGateway.
Feature: 004-concurrency-gateway-sessions (Tasks T009, T014)
"""

import pytest
import asyncio
import time
from fastapi import HTTPException

from app.core.gateway import ConcurrencyGateway
from app.core.telemetry import OperationalMetricsTracker


@pytest.mark.asyncio
async def test_semaphore_concurrency_slots():
    """Verify that ConcurrencyGateway allows exactly up to N concurrent slots."""
    tracker = OperationalMetricsTracker(concurrency_limit=5)
    gw = ConcurrencyGateway(concurrency_limit=5, queue_timeout_seconds=2.0, tracker=tracker)

    active_tasks = 0
    max_active = 0
    lock = asyncio.Lock()

    async def worker():
        nonlocal active_tasks, max_active
        async with gw.acquire_slot() as queue_wait_ms:
            async with lock:
                active_tasks += 1
                if active_tasks > max_active:
                    max_active = active_tasks
            await asyncio.sleep(0.05)
            async with lock:
                active_tasks -= 1

    # Launch 10 workers against a concurrency limit of 5
    await asyncio.gather(*[worker() for _ in range(10)])

    assert max_active == 5  # Strictly enforced capacity ceiling
    assert active_tasks == 0


@pytest.mark.asyncio
async def test_semaphore_queue_wait_tracking():
    """Verify that queue wait time is tracked when workers must queue for slots."""
    tracker = OperationalMetricsTracker(concurrency_limit=1)
    gw = ConcurrencyGateway(concurrency_limit=1, queue_timeout_seconds=2.0, tracker=tracker)

    queue_times = []

    async def worker(sleep_time: float):
        async with gw.acquire_slot() as queue_wait_ms:
            queue_times.append(queue_wait_ms)
            await asyncio.sleep(sleep_time)

    # Worker 1 holds the slot for 0.1s; Worker 2 must wait
    await asyncio.gather(worker(0.1), worker(0.01))

    assert len(queue_times) == 2
    assert queue_times[0] < 20.0       # First acquired immediately
    assert queue_times[1] >= 90.0      # Second waited ~100ms


@pytest.mark.asyncio
async def test_semaphore_timeout_rate_limit_exceeded():
    """Verify that when queue timeout is exceeded, HTTP 429 is raised."""
    tracker = OperationalMetricsTracker(concurrency_limit=1)
    # Set tight timeout of 0.05s
    gw = ConcurrencyGateway(concurrency_limit=1, queue_timeout_seconds=0.05, tracker=tracker)

    async def blocking_worker():
        async with gw.acquire_slot():
            await asyncio.sleep(0.2)

    async def failing_worker():
        await asyncio.sleep(0.01)
        with pytest.raises(HTTPException) as exc_info:
            async with gw.acquire_slot():
                pass
        assert exc_info.value.status_code == 429
        assert "Retry-After" in exc_info.value.headers

    await asyncio.gather(blocking_worker(), failing_worker())


@pytest.mark.asyncio
async def test_50_parallel_requests_e2e():
    """
    Simulate 50 simultaneous users acquiring gateway concurrency slots.
    Asserts zero exceptions, zero connection starvation, and 100% completion.
    """
    tracker = OperationalMetricsTracker(concurrency_limit=50)
    gw = ConcurrencyGateway(concurrency_limit=50, queue_timeout_seconds=5.0, tracker=tracker)

    completed = 0
    lock = asyncio.Lock()

    async def simulate_user(user_id: int):
        nonlocal completed
        t0 = time.perf_counter()
        async with gw.acquire_slot() as queue_wait_ms:
            # Simulate sub-second inference task
            await asyncio.sleep(0.02)
            async with lock:
                completed += 1

    # Launch 50 simultaneous virtual users
    tasks = [simulate_user(i) for i in range(50)]
    t_start = time.perf_counter()
    await asyncio.gather(*tasks)
    total_elapsed = time.perf_counter() - t_start

    assert completed == 50
    assert total_elapsed < 2.0  # Finished within p95 latency envelope
