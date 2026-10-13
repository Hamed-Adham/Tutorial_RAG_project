"""
Pytest fixtures for backend gateway and session tests.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, AsyncMock

from app.main import app
from app.services.session_store import InMemorySessionStore
from app.schemas.chat import ChatResponse
from rag_core.indexing.science_schema import PageCitation


@pytest_asyncio.fixture
async def async_client():
    """Asynchronous HTTP test client bound to FastAPI application."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def in_memory_store():
    """Clean in-memory session store fixture."""
    return InMemorySessionStore(default_ttl_seconds=3600, max_turns=4)


@pytest.fixture
def mock_chat_response():
    """Mock ChatResponse fixture."""
    return ChatResponse(
        session_id="sess_test_123",
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
        trace_id="0194e8bc-mock"
    )
