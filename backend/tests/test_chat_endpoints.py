"""
Integration Tests for FastAPI Endpoints: Chat, Stream (SSE), Health, Metrics, Sessions.
Feature: 004-concurrency-gateway-sessions (Tasks T020, T023)
"""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock
from httpx import AsyncClient

from app.schemas.chat import ChatResponse
from app.services.rag_service import rag_service
from rag_core.indexing.science_schema import PageCitation


@pytest.mark.asyncio
async def test_root_status_endpoint(async_client: AsyncClient):
    """GET / must return 200 with service info and endpoint links."""
    resp = await async_client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "service" in data
    assert data["docs"] == "/docs"


@pytest.mark.asyncio
async def test_health_endpoint(async_client: AsyncClient):
    """GET /api/v1/health must return readiness probe with component statuses."""
    resp = await async_client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded")
    assert "qdrant_connected" in data
    assert "bm25_loaded" in data
    assert "uptime_seconds" in data


@pytest.mark.asyncio
async def test_metrics_endpoint(async_client: AsyncClient):
    """GET /api/v1/metrics must return live operational KPIs."""
    resp = await async_client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "active_requests" in data
    assert "available_semaphore_slots" in data
    assert "circuit_breaker_trip_rate" in data
    assert "avg_queue_wait_ms" in data
    assert "latency_p95_ms" in data


@pytest.mark.asyncio
async def test_batch_chat_endpoint_with_mock(async_client: AsyncClient, monkeypatch):
    """POST /api/v1/chat must return a structured ChatResponse."""
    mock_resp = ChatResponse(
        session_id="sess_mock_42",
        query="فرمول چگالی چیست؟",
        effective_query="فرمول چگالی چیست؟",
        answer="چگالی برابر است با جرم تقسیم بر حجم (d = m / V).",
        citations=[
            PageCitation(
                doc_id="C706_olom_haftom-7-144-6-11.pdf",
                filename="کتاب علوم تجربی هفتم",
                page=4,
                chunk_id="SEC-0015_c0",
                score=0.92
            )
        ],
        is_refusal=False,
        latency_profile={"queue_wait_ms": 1.2, "total_ms": 1150.0},
        trace_id="trace_abc123"
    )

    async def mock_execute(req):
        return mock_resp

    monkeypatch.setattr(rag_service, "execute_chat", mock_execute)

    payload = {
        "query": "فرمول چگالی چیست؟",
        "session_id": "sess_mock_42",
        "selected_chapters": [2],
        "top_k": 5
    }
    resp = await async_client.post("/api/v1/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["session_id"] == "sess_mock_42"
    assert "چگالی" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["page"] == 4
    assert data["trace_id"] == "trace_abc123"


@pytest.mark.asyncio
async def test_streaming_chat_endpoint_sse(async_client: AsyncClient, monkeypatch):
    """POST /api/v1/chat/stream must return text/event-stream with token and done events."""
    async def mock_stream(req):
        tokens = ["چگالی ", "برابر ", "است با جرم تقسیم بر حجم."]
        for i, t in enumerate(tokens, 1):
            yield f'event: token\ndata: {{"content": {json.dumps(t)}, "seq": {i}}}\n\n'
        done = {
            "session_id": req.session_id or "sess_stream",
            "effective_query": req.query,
            "ttft_ms": 142.5,
            "total_ms": 1120.0
        }
        yield f'event: done\ndata: {json.dumps(done)}\n\n'

    monkeypatch.setattr(rag_service, "execute_chat_stream", mock_stream)

    payload = {
        "query": "فرمول چگالی چیست؟",
        "session_id": "sess_stream_01",
        "stream": True
    }
    resp = await async_client.post("/api/v1/chat/stream", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    body = resp.text
    assert "event: token" in body
    assert "event: done" in body
    assert "ttft_ms" in body


@pytest.mark.asyncio
async def test_session_lifecycle_endpoint(async_client: AsyncClient):
    """DELETE /api/v1/sessions/{session_id} deletes session."""
    resp = await async_client.delete("/api/v1/sessions/nonexistent_session_123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "nonexistent_session_123"
    assert data["deleted"] is False
