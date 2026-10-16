"""
Session Store implementation: ISessionStore, InMemorySessionStore, and RedisSessionStore.
Provides sliding-window (last 4 dialogue turns) retention and 2-hour TTL expiration.
"""

import json
import uuid
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from rag_core.indexing.science_schema import PageCitation
from app.schemas.session import SessionStateModel, DialogueTurnModel
from app.core.config import settings

logger = logging.getLogger(__name__)


class ISessionStore(ABC):
    """Protocol for conversation session storage."""

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[SessionStateModel]:
        """Fetch session state by ID or return None if expired / absent."""
        pass

    @abstractmethod
    async def save_session(self, session: SessionStateModel) -> None:
        """Persist or update session state and refresh expiration TTL."""
        pass

    @abstractmethod
    async def append_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        citations: List[PageCitation],
        latencies: Dict[str, float],
        selected_chapters: Optional[List[int]] = None,
        q_and_a_summary: Optional[str] = None,
        turn_metadata: Optional[Dict[str, Any]] = None
    ) -> SessionStateModel:
        """Append a user+assistant turn pair, enforce sliding window, and reset TTL."""
        pass

    @abstractmethod
    async def update_turn_summary(self, session_id: str, turn_id: str, summary: str) -> bool:
        """Update q_and_a_summary on a specific turn."""
        pass

    @abstractmethod
    async def batch_update_summaries(self, session_id: str, summaries: Dict[str, str]) -> bool:
        """Batch update q_and_a_summary across multiple turns in a session."""
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Explicitly purge session state."""
        pass


class InMemorySessionStore(ISessionStore):
    """
    Thread-safe in-memory session store fallback with TTL eviction.
    Operates with zero external dependencies for offline development and testing.
    """

    def __init__(self, default_ttl_seconds: int = settings.session_ttl_seconds, max_turns: int = settings.max_history_turns):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._default_ttl = default_ttl_seconds
        self._max_turns = max_turns
        self._lock = asyncio.Lock()

    async def get_session(self, session_id: str) -> Optional[SessionStateModel]:
        async with self._lock:
            data = self._store.get(session_id)
            if not data:
                return None

            now = datetime.now(timezone.utc)
            last_active = data["session"].last_active_at
            if (now - last_active).total_seconds() > data["ttl"]:
                # Evict expired session
                del self._store[session_id]
                return None

            return data["session"]

    async def save_session(self, session: SessionStateModel) -> None:
        async with self._lock:
            self._store[session.session_id] = {
                "session": session,
                "ttl": session.ttl_seconds
            }

    async def append_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        citations: List[PageCitation],
        latencies: Dict[str, float],
        selected_chapters: Optional[List[int]] = None,
        q_and_a_summary: Optional[str] = None,
        turn_metadata: Optional[Dict[str, Any]] = None
    ) -> SessionStateModel:
        session = await self.get_session(session_id)
        now = datetime.now(timezone.utc)

        if not session:
            session = SessionStateModel(
                session_id=session_id,
                created_at=now,
                last_active_at=now,
                turns=[],
                selected_chapters=selected_chapters,
                ttl_seconds=self._default_ttl
            )

        turn_pair_id = f"turn_{uuid.uuid4().hex[:12]}"

        # Append user turn
        session.turns.append(
            DialogueTurnModel(
                turn_id=str(uuid.uuid4()),
                turn_pair_id=turn_pair_id,
                role="user",
                content=user_query,
                timestamp=now,
                citations=[],
                latency_profile={},
                metadata={"turn_pair_id": turn_pair_id}
            )
        )
        # Prepare assistant metadata
        asst_metadata = dict(turn_metadata or {})
        asst_metadata["turn_pair_id"] = turn_pair_id
        if q_and_a_summary:
            asst_metadata["q_and_a_summary"] = q_and_a_summary

        # Append assistant turn
        session.turns.append(
            DialogueTurnModel(
                turn_id=str(uuid.uuid4()),
                turn_pair_id=turn_pair_id,
                role="assistant",
                content=assistant_answer,
                timestamp=now,
                citations=citations,
                latency_profile=latencies,
                q_and_a_summary=q_and_a_summary,
                metadata=asst_metadata
            )
        )

        # Enforce sliding window (keep last max_turns turns)
        if len(session.turns) > self._max_turns:
            session.turns = session.turns[-self._max_turns:]

        session.last_active_at = now
        if selected_chapters is not None:
            session.selected_chapters = selected_chapters

        await self.save_session(session)
        return session

    async def update_turn_summary(self, session_id: str, turn_id: str, summary: str) -> bool:
        session = await self.get_session(session_id)
        if not session:
            return False
        updated = False
        for turn in session.turns:
            if turn.turn_id == turn_id or getattr(turn, "turn_pair_id", None) == turn_id:
                turn.metadata["q_and_a_summary"] = summary
                updated = True
                if turn.role == "assistant":
                    break
        if updated:
            await self.save_session(session)
        return updated

    async def batch_update_summaries(self, session_id: str, summaries: Dict[str, str]) -> bool:
        session = await self.get_session(session_id)
        if not session:
            return False
        updated = False
        for turn in session.turns:
            tid = turn.turn_id
            tpid = getattr(turn, "turn_pair_id", None)
            if tid in summaries:
                turn.metadata["q_and_a_summary"] = summaries[tid]
                updated = True
            elif tpid and tpid in summaries:
                turn.metadata["q_and_a_summary"] = summaries[tpid]
                updated = True
        if updated:
            await self.save_session(session)
        return updated

    async def delete_session(self, session_id: str) -> bool:
        async with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                return True
            return False


class RedisSessionStore(ISessionStore):
    """
    High-performance Redis session store using redis.asyncio.
    Automatically degrades to InMemorySessionStore fallback if Redis is unavailable.
    """

    def __init__(
        self,
        redis_url: str = settings.redis_url,
        ttl_seconds: int = settings.session_ttl_seconds,
        max_turns: int = settings.max_history_turns
    ):
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self._fallback = InMemorySessionStore(default_ttl_seconds=ttl_seconds, max_turns=max_turns)
        self._redis = None
        self._redis_available = False

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=2.0
                )
                await self._redis.ping()
                self._redis_available = True
                logger.info("Connected successfully to Redis session store at %s", self.redis_url)
            except Exception as e:
                self._redis_available = False
                logger.warning("Redis unavailable (%s), falling back to in-memory session store", e)
        return self._redis

    async def get_session(self, session_id: str) -> Optional[SessionStateModel]:
        client = await self._get_redis()
        if not self._redis_available or client is None:
            return await self._fallback.get_session(session_id)

        try:
            raw = await client.get(f"session:{session_id}")
            if not raw:
                return None
            data = json.loads(raw)
            return SessionStateModel.model_validate(data)
        except Exception as e:
            logger.warning("Redis get_session failed (%s), using fallback", e)
            return await self._fallback.get_session(session_id)

    async def save_session(self, session: SessionStateModel) -> None:
        client = await self._get_redis()
        if not self._redis_available or client is None:
            await self._fallback.save_session(session)
            return

        try:
            payload = session.model_dump_json()
            await client.setex(f"session:{session.session_id}", self.ttl_seconds, payload)
        except Exception as e:
            logger.warning("Redis save_session failed (%s), using fallback", e)
            await self._fallback.save_session(session)

    async def append_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        citations: List[PageCitation],
        latencies: Dict[str, float],
        selected_chapters: Optional[List[int]] = None,
        q_and_a_summary: Optional[str] = None,
        turn_metadata: Optional[Dict[str, Any]] = None
    ) -> SessionStateModel:
        session = await self.get_session(session_id)
        now = datetime.now(timezone.utc)

        if not session:
            session = SessionStateModel(
                session_id=session_id,
                created_at=now,
                last_active_at=now,
                turns=[],
                selected_chapters=selected_chapters,
                ttl_seconds=self.ttl_seconds
            )

        turn_pair_id = f"turn_{uuid.uuid4().hex[:12]}"

        session.turns.append(
            DialogueTurnModel(
                turn_id=str(uuid.uuid4()),
                turn_pair_id=turn_pair_id,
                role="user",
                content=user_query,
                timestamp=now,
                citations=[],
                latency_profile={},
                metadata={"turn_pair_id": turn_pair_id}
            )
        )

        asst_metadata = dict(turn_metadata or {})
        asst_metadata["turn_pair_id"] = turn_pair_id
        if q_and_a_summary:
            asst_metadata["q_and_a_summary"] = q_and_a_summary

        session.turns.append(
            DialogueTurnModel(
                turn_id=str(uuid.uuid4()),
                turn_pair_id=turn_pair_id,
                role="assistant",
                content=assistant_answer,
                timestamp=now,
                citations=citations,
                latency_profile=latencies,
                q_and_a_summary=q_and_a_summary,
                metadata=asst_metadata
            )
        )

        if len(session.turns) > self.max_turns:
            session.turns = session.turns[-self.max_turns:]

        session.last_active_at = now
        if selected_chapters is not None:
            session.selected_chapters = selected_chapters

        await self.save_session(session)
        return session

    async def update_turn_summary(self, session_id: str, turn_id: str, summary: str) -> bool:
        client = await self._get_redis()
        if not self._redis_available or client is None:
            return await self._fallback.update_turn_summary(session_id, turn_id, summary)

        session = await self.get_session(session_id)
        if not session:
            return False
        updated = False
        for turn in session.turns:
            if turn.turn_id == turn_id or getattr(turn, "turn_pair_id", None) == turn_id:
                turn.metadata["q_and_a_summary"] = summary
                updated = True
                if turn.role == "assistant":
                    break
        if updated:
            await self.save_session(session)
        return updated

    async def batch_update_summaries(self, session_id: str, summaries: Dict[str, str]) -> bool:
        client = await self._get_redis()
        if not self._redis_available or client is None:
            return await self._fallback.batch_update_summaries(session_id, summaries)

        session = await self.get_session(session_id)
        if not session:
            return False
        updated = False
        for turn in session.turns:
            tid = turn.turn_id
            tpid = getattr(turn, "turn_pair_id", None)
            if tid in summaries:
                turn.metadata["q_and_a_summary"] = summaries[tid]
                updated = True
            elif tpid and tpid in summaries:
                turn.metadata["q_and_a_summary"] = summaries[tpid]
                updated = True
        if updated:
            await self.save_session(session)
        return updated

    async def delete_session(self, session_id: str) -> bool:
        client = await self._get_redis()
        if not self._redis_available or client is None:
            return await self._fallback.delete_session(session_id)

        try:
            res = await client.delete(f"session:{session_id}")
            return bool(res > 0)
        except Exception as e:
            logger.warning("Redis delete_session failed (%s), using fallback", e)
            return await self._fallback.delete_session(session_id)


# Global singleton instance
_session_store_instance: Optional[ISessionStore] = None

def get_session_store() -> ISessionStore:
    """Return the configured global session store singleton."""
    global _session_store_instance
    if _session_store_instance is None:
        if settings.use_redis:
            _session_store_instance = RedisSessionStore()
        else:
            _session_store_instance = InMemorySessionStore()
    return _session_store_instance
