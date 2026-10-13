"""
Concurrency Gateway: Implements asyncio.Semaphore rate-guarding and queue latency tracking.
Protects the inference and retrieval pipeline from resource exhaustion during load spikes.
"""

import time
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from fastapi import HTTPException

from app.core.config import settings
from app.core.telemetry import metrics_tracker

logger = logging.getLogger(__name__)


class ConcurrencyGateway:
    """
    Manages global request concurrency using an asynchronous semaphore.
    Enforces maximum concurrent execution limits (default 50) and tracks queue delay.
    """

    def __init__(
        self,
        concurrency_limit: int = settings.concurrency_limit,
        queue_timeout_seconds: float = settings.semaphore_timeout_seconds,
        tracker = metrics_tracker
    ):
        self.concurrency_limit = concurrency_limit
        self.queue_timeout_seconds = queue_timeout_seconds
        self.tracker = tracker
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._lock = asyncio.Lock()

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazily initialize the semaphore within the running event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency_limit)
        return self._semaphore

    @asynccontextmanager
    async def acquire_slot(self) -> AsyncIterator[float]:
        """
        Asynchronously acquires a concurrency slot.
        Yields the queue wait latency in milliseconds.
        Raises HTTP 429 if queue wait exceeds queue_timeout_seconds.
        """
        sem = self._get_semaphore()
        t_request = time.perf_counter()
        acquired = False

        try:
            # Attempt to acquire slot within timeout budget
            try:
                await asyncio.wait_for(sem.acquire(), timeout=self.queue_timeout_seconds)
                acquired = True
            except asyncio.TimeoutError:
                queue_wait_ms = (time.perf_counter() - t_request) * 1000.0
                logger.warning(
                    "Concurrency rate limit exceeded (waited %.2f ms, limit=%d)",
                    queue_wait_ms,
                    self.concurrency_limit
                )
                raise HTTPException(
                    status_code=429,
                    detail="System is experiencing peak concurrent load. Please retry in a few moments.",
                    headers={"Retry-After": "5"}
                )

            t_acquired = time.perf_counter()
            queue_wait_ms = round((t_acquired - t_request) * 1000.0, 2)
            yield queue_wait_ms

        finally:
            if acquired:
                sem.release()


# Global concurrency gateway singleton
gateway = ConcurrencyGateway()
