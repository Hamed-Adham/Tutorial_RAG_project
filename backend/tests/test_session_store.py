"""
Unit & Integration Tests for ISessionStore, InMemorySessionStore, and RedisSessionStore.
Feature: 004-concurrency-gateway-sessions (Tasks T015, T019)
"""

import pytest
import asyncio
from datetime import datetime, timezone

from app.services.session_store import InMemorySessionStore, RedisSessionStore
from rag_core.indexing.science_schema import PageCitation


@pytest.mark.asyncio
async def test_in_memory_session_store_lifecycle():
    """Verify session creation, turn appending, retrieval, and deletion."""
    store = InMemorySessionStore(default_ttl_seconds=3600, max_turns=4)

    # Initially absent
    res = await store.get_session("sess_alpha")
    assert res is None

    # Append first turn pair
    citations = [
        PageCitation(doc_id="doc1.pdf", filename="کتاب علوم", page=4, chunk_id="c1", score=0.9)
    ]
    s = await store.append_turn(
        session_id="sess_alpha",
        user_query="استوانه مدرج چیست؟",
        assistant_answer="ابزاری برای اندازه گیری حجم مایعات است.",
        citations=citations,
        latencies={"retrieval_ms": 12.0, "total_ms": 150.0},
        selected_chapters=[1, 2]
    )

    assert s.session_id == "sess_alpha"
    assert len(s.turns) == 2
    assert s.turns[0].role == "user"
    assert s.turns[0].content == "استوانه مدرج چیست؟"
    assert s.turns[1].role == "assistant"
    assert len(s.turns[1].citations) == 1
    assert s.selected_chapters == [1, 2]

    # Verify shared turn_pair_id linking Q&A
    assert s.turns[0].turn_pair_id is not None
    assert s.turns[0].turn_pair_id.startswith("turn_")
    assert s.turns[0].turn_pair_id == s.turns[1].turn_pair_id

    # Retrieve and verify
    fetched = await store.get_session("sess_alpha")
    assert fetched is not None
    assert len(fetched.turns) == 2

    # Update summary by turn_pair_id
    updated = await store.update_turn_summary("sess_alpha", s.turns[0].turn_pair_id, "خلاصه اندازه‌گیری حجم")
    assert updated is True
    fetched_after = await store.get_session("sess_alpha")
    assert fetched_after.turns[1].metadata.get("q_and_a_summary") == "خلاصه اندازه‌گیری حجم"

    # Delete session
    deleted = await store.delete_session("sess_alpha")
    assert deleted is True
    assert await store.get_session("sess_alpha") is None


@pytest.mark.asyncio
async def test_sliding_window_context_cap():
    """Verify that conversation history is strictly capped at max_turns (last 4 turns)."""
    store = InMemorySessionStore(default_ttl_seconds=3600, max_turns=4)

    # Append 3 turn pairs (total 6 turns)
    await store.append_turn("sess_window", "پرسش ۱", "پاسخ ۱", [], {})
    await store.append_turn("sess_window", "پرسش ۲", "پاسخ ۲", [], {})
    final_session = await store.append_turn("sess_window", "پرسش ۳", "پاسخ ۳", [], {})

    assert len(final_session.turns) == 4
    # Oldest pair (پرسش ۱, پاسخ ۱) should have been evicted
    contents = [t.content for t in final_session.turns]
    assert "پرسش ۱" not in contents
    assert "پاسخ ۱" not in contents
    assert contents == ["پرسش ۲", "پاسخ ۲", "پرسش ۳", "پاسخ ۳"]


@pytest.mark.asyncio
async def test_session_store_ttl_expiration():
    """Verify that sessions expire and evict cleanly after TTL window."""
    # Set tight TTL of 0.05 seconds
    store = InMemorySessionStore(default_ttl_seconds=0.05, max_turns=4)

    await store.append_turn("sess_expire", "سلام", "درود!", [], {})
    assert await store.get_session("sess_expire") is not None

    # Wait for TTL expiration
    await asyncio.sleep(0.08)
    assert await store.get_session("sess_expire") is None


@pytest.mark.asyncio
async def test_multi_tenant_session_isolation():
    """Verify strict tenant isolation between different session IDs."""
    store = InMemorySessionStore(default_ttl_seconds=3600, max_turns=4)

    await store.append_turn("student_A", "سوال دانش‌آموز الف", "پاسخ الف", [], {})
    await store.append_turn("student_B", "سوال دانش‌آموز ب", "پاسخ ب", [], {})

    session_a = await store.get_session("student_A")
    session_b = await store.get_session("student_B")

    assert session_a.turns[0].content == "سوال دانش‌آموز الف"
    assert session_b.turns[0].content == "سوال دانش‌آموز ب"
    assert len(session_a.turns) == 2
    assert len(session_b.turns) == 2


@pytest.mark.asyncio
async def test_redis_session_store_fallback_when_offline():
    """Verify that RedisSessionStore falls back seamlessly to in-memory store when Redis is offline."""
    # Invalid Redis port forces connection error
    store = RedisSessionStore(redis_url="redis://localhost:9999/0", ttl_seconds=3600, max_turns=4)

    s = await store.append_turn("sess_fallback", "پرسش آزمایشی", "پاسخ آزمایشی", [], {})
    assert s is not None
    assert len(s.turns) == 2

    fetched = await store.get_session("sess_fallback")
    assert fetched is not None
    assert fetched.turns[0].content == "پرسش آزمایشی"

    deleted = await store.delete_session("sess_fallback")
    assert deleted is True
    assert await store.get_session("sess_fallback") is None
