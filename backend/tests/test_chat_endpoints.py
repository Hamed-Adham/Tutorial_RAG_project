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


@pytest.mark.asyncio
async def test_rag_service_langgraph_engine_routing_batch(monkeypatch):
    """Verify that RAGService routes through EnterpriseScienceGraph.ainvoke when use_langgraph_engine is True."""
    from app.core.config import settings
    from app.schemas.chat import ChatRequest

    monkeypatch.setattr(settings, "use_langgraph_engine", True)

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "agent_response": "پاسخ پردازش شده توسط ارکستریتور LangGraph",
        "effective_query": "فرمول چگالی چیست؟",
        "is_refusal": False,
        "citations": [
            {
                "doc_id": "science_textbook.pdf",
                "filename": "کتاب علوم تجربی هفتم",
                "page": 4,
                "score": 0.94,
                "chunk_id": "chk_01"
            }
        ],
        "latencies": {"rewrite_ms": 15.0, "synthesis_ms": 110.0},
        "trace_id": "trace_lg_test"
    })

    monkeypatch.setattr(rag_service, "_enterprise_graph", mock_graph)

    req = ChatRequest(query="فرمول چگالی چیست؟", session_id="sess_lg_test_batch")
    resp = await rag_service.execute_chat(req)

    mock_graph.ainvoke.assert_called_once()
    assert "LangGraph" in resp.answer
    assert resp.effective_query == "فرمول چگالی چیست؟"
    assert len(resp.citations) == 1
    assert resp.citations[0].page == 4
    assert resp.trace_id == "trace_lg_test"
    assert resp.is_refusal is False


@pytest.mark.asyncio
async def test_rag_service_langgraph_engine_routing_stream(monkeypatch):
    """Verify that RAGService routes through EnterpriseScienceGraph.astream_tokens when use_langgraph_engine is True."""
    from app.core.config import settings
    from app.schemas.chat import ChatRequest

    monkeypatch.setattr(settings, "use_langgraph_engine", True)

    async def mock_astream_tokens(**kwargs):
        for t in ["جریان ", "توکن‌های ", "LangGraph"]:
            yield t

    mock_graph = MagicMock()
    mock_graph.astream_tokens = mock_astream_tokens
    mock_graph.get_citations_for_session.return_value = [
        PageCitation(
            doc_id="science_textbook.pdf",
            filename="کتاب علوم تجربی هفتم",
            page=4,
            score=0.91,
            chunk_id="chk_stream_01"
        )
    ]
    mock_graph.get_refusal_for_session.return_value = False
    mock_graph.get_effective_query_for_session.return_value = "فرمول چگالی چیست؟"
    mock_graph.last_ttft_ms = 85.0

    monkeypatch.setattr(rag_service, "_enterprise_graph", mock_graph)

    req = ChatRequest(query="فرمول چگالی چیست؟", session_id="sess_lg_test_stream", stream=True)
    events = []
    async for frame in rag_service.execute_chat_stream(req):
        events.append(frame)

    full_frames = "".join(events)
    assert "event: token" in full_frames
    assert "جریان" in full_frames
    assert "LangGraph" in full_frames
    assert "event: done" in full_frames
    assert "ttft_ms" in full_frames


# ---------------------------------------------------------------------------
# Feature 006 Quickstart Scenarios 1-6 End-to-End Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_1_completely_irrelevant_gating(async_client: AsyncClient, monkeypatch):
    """Scenario 1: Completely irrelevant gating results in instant refusal (<25ms) and empty citations."""
    mock_resp = ChatResponse(
        session_id="sess_sc1",
        query="نظرت درباره بازی استقلال و پرسپولیس چیه؟",
        effective_query="نظرت درباره بازی استقلال و پرسپولیس چیه؟",
        answer="این پرسش خارج از حیطه کتاب درسی علوم تجربی پایه هفتم است.",
        citations=[],
        is_refusal=True,
        intent_tier="completely_irrelevant",
        context_need_tier="needs_neither",
        latency_profile={"rewrite_ms": 12.0, "retrieval_ms": 0.0, "total_ms": 15.0}
    )

    async def mock_execute(req):
        return mock_resp

    monkeypatch.setattr(rag_service, "execute_chat", mock_execute)

    resp = await async_client.post("/api/v1/chat", json={
        "query": "نظرت درباره بازی استقلال و پرسپولیس چیه؟",
        "session_id": "sess_sc1"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_refusal"] is True
    assert data["intent_tier"] == "completely_irrelevant"
    assert len(data["citations"]) == 0


@pytest.mark.asyncio
async def test_scenario_2_partially_relevant_distillation(async_client: AsyncClient, monkeypatch):
    """Scenario 2: Partially relevant inquiry has chatter stripped and core inquiry distilled."""
    mock_resp = ChatResponse(
        session_id="sess_sc2",
        query="امروز هوا بارونیه ولی بگو چگالی چوب چقدره؟",
        effective_query="چگالی چوب چقدر است؟",
        answer="چگالی چوب بسته به نوع آن متغیر و کمتر از آب است.",
        citations=[
            PageCitation(doc_id="olom.pdf", filename="علوم هفتم", page=6, score=0.89)
        ],
        is_refusal=False,
        intent_tier="partially_relevant",
        context_need_tier="needs_retrieve_only",
        latency_profile={"rewrite_ms": 45.0, "total_ms": 820.0}
    )

    async def mock_execute(req):
        return mock_resp

    monkeypatch.setattr(rag_service, "execute_chat", mock_execute)

    resp = await async_client.post("/api/v1/chat", json={
        "query": "امروز هوا بارونیه ولی بگو چگالی چوب چقدره؟",
        "session_id": "sess_sc2"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent_tier"] == "partially_relevant"
    assert data["effective_query"] == "چگالی چوب چقدر است؟"
    assert len(data["citations"]) == 1


@pytest.mark.asyncio
async def test_scenario_3_needs_history_only_badge(async_client: AsyncClient, monkeypatch):
    """Scenario 3: Meta-inquiry bypasses retrieval, returns conversational context badge and empty citations."""
    mock_resp = ChatResponse(
        session_id="sess_sc3",
        query="پاسخ قبلی را خلاصه کن",
        effective_query="پاسخ قبلی را خلاصه کن",
        answer="خلاصه پاسخ پیشین: آبخوان‌ها لایه‌های آبدار زیرزمینی هستند.",
        citations=[],
        is_refusal=False,
        conversational_context_badge="پاسخ بر اساس گفتگوی پیشین",
        intent_tier="relevant",
        context_need_tier="needs_history_only",
        latency_profile={"rewrite_ms": 15.0, "retrieval_ms": 0.0, "synthesis_ms": 180.0, "total_ms": 195.0}
    )

    async def mock_execute(req):
        return mock_resp

    monkeypatch.setattr(rag_service, "execute_chat", mock_execute)

    resp = await async_client.post("/api/v1/chat", json={
        "query": "پاسخ قبلی را خلاصه کن",
        "session_id": "sess_sc3"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversational_context_badge"] == "پاسخ بر اساس گفتگوی پیشین"
    assert data["context_need_tier"] == "needs_history_only"
    assert len(data["citations"]) == 0


@pytest.mark.asyncio
async def test_scenario_4_subq_decision_model_2(async_client: AsyncClient, monkeypatch):
    """Scenario 4: Multi-turn inquiry requiring both history and retrieval dispatches via Decision Model 2."""
    mock_resp = ChatResponse(
        session_id="sess_sc4",
        query="آزمایشی که توضیح دادی چه خطاهایی دارد؟",
        effective_query="خطاهای آزمایش اندازه‌گیری چگالی چیست؟",
        answer="خطاهای این آزمایش شامل خطای دید در استوانه مدرج و خطای صفر ترازو است.",
        citations=[
            PageCitation(doc_id="olom.pdf", filename="علوم هفتم", page=8, score=0.93)
        ],
        is_refusal=False,
        intent_tier="relevant",
        context_need_tier="needs_history_and_retrieve",
        latency_profile={"rewrite_ms": 210.0, "retrieval_ms": 40.0, "total_ms": 950.0}
    )

    async def mock_execute(req):
        return mock_resp

    monkeypatch.setattr(rag_service, "execute_chat", mock_execute)

    resp = await async_client.post("/api/v1/chat", json={
        "query": "آزمایشی که توضیح دادی چه خطاهایی دارد؟",
        "session_id": "sess_sc4"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["context_need_tier"] == "needs_history_and_retrieve"
    assert len(data["citations"]) == 1


@pytest.mark.asyncio
async def test_scenario_5_ui_reply_and_manual_context(async_client: AsyncClient, monkeypatch):
    """Scenario 5: User passes selected_context_turn_id when history is OFF, bypassing Model 2."""
    captured_req = []

    async def mock_execute(req):
        captured_req.append(req)
        return ChatResponse(
            session_id=req.session_id or "sess_sc5",
            query=req.query,
            effective_query=req.query,
            answer="پاسخ مستقیم بر اساس بافت انتخاب‌شده.",
            citations=[],
            is_refusal=False,
            conversational_context_badge="پاسخ بر اساس پیام انتخاب‌شده",
            intent_tier="relevant",
            context_need_tier="needs_history_only"
        )

    monkeypatch.setattr(rag_service, "execute_chat", mock_execute)

    payload = {
        "query": "این بخش را بیشتر باز کن",
        "session_id": "sess_sc5",
        "history_enabled": False,
        "selected_context_turn_id": "turn_past_3"
    }
    resp = await async_client.post("/api/v1/chat", json=payload)
    assert resp.status_code == 200
    assert len(captured_req) == 1
    assert captured_req[0].history_enabled is False
    assert captured_req[0].selected_context_turn_id == "turn_past_3"


@pytest.mark.asyncio
async def test_scenario_6_turn_0_cold_start_coercion(async_client: AsyncClient, monkeypatch):
    """Scenario 6: Turn 0 inquiry with no history is coerced from needs_history_and_retrieve to needs_retrieve_only."""
    mock_resp = ChatResponse(
        session_id="sess_sc6",
        query="چگونه آبخوان تشکیل می‌شود؟",
        effective_query="چگونه آبخوان تشکیل می‌شود؟",
        answer="آبخوان بر اثر نفوذ آب باران در لایه‌های نفوذپذیر زمین تشکیل می‌شود.",
        citations=[
            PageCitation(doc_id="olom.pdf", filename="علوم هفتم", page=56, score=0.91)
        ],
        is_refusal=False,
        intent_tier="relevant",
        context_need_tier="needs_retrieve_only"
    )

    async def mock_execute(req):
        return mock_resp

    monkeypatch.setattr(rag_service, "execute_chat", mock_execute)

    resp = await async_client.post("/api/v1/chat", json={
        "query": "چگونه آبخوان تشکیل می‌شود؟",
        "session_id": "sess_sc6",
        "history": []
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["context_need_tier"] == "needs_retrieve_only"
    assert data["is_refusal"] is False


