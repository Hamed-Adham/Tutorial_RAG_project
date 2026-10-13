"""
Operational Metrics Endpoint: Real-time production KPIs for monitoring and autoscaling.
"""

from fastapi import APIRouter
from app.contracts.gateway_contracts import OperationalMetricsModel
from app.core.telemetry import metrics_tracker

router = APIRouter()


@router.get("/metrics", response_model=OperationalMetricsModel, summary="Operational KPIs & Telemetry")
async def metrics_endpoint() -> OperationalMetricsModel:
    """Returns active requests, semaphore queue wait latency, circuit breaker trip rate, and latency percentiles."""
    return metrics_tracker.get_metrics_snapshot()
