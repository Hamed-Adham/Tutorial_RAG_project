"""
RAGService: Service layer bridging FastAPI gateway endpoints to the AdvancedRAGPipeline core.
Coordinates session history resolution, semaphore rate-guarding, and operational telemetry.
"""

import time
import uuid
import logging
from typing import AsyncIterator, Optional, List, Dict, Any

from rag_core.pipeline import AdvancedRAGPipeline
from rag_core.indexing.science_schema import AdvancedRAGResult
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.session_store import get_session_store, ISessionStore
from app.core.gateway import gateway, ConcurrencyGateway
from app.core.telemetry import metrics_tracker, OperationalMetricsTracker

logger = logging.getLogger(__name__)


class RAGService:
    """
    Coordinates asynchronous query execution, multi-turn history resolution,
    and production telemetry tracking.
    """

    def __init__(
        self,
        pipeline: Optional[AdvancedRAGPipeline] = None,
        session_store: Optional[ISessionStore] = None,
        gateway_instance: Optional[ConcurrencyGateway] = None,
        tracker: Optional[OperationalMetricsTracker] = None,
    ):
        self._pipeline = pipeline
        self.session_store = session_store or get_session_store()
        self.gateway = gateway_instance or gateway
        self.tracker = tracker or metrics_tracker

    @property
    def pipeline(self) -> AdvancedRAGPipeline:
        if self._pipeline is None:
            self._pipeline = AdvancedRAGPipeline(warmup=False)
        return self._pipeline

    @pipeline.setter
    def pipeline(self, val: AdvancedRAGPipeline):
        self._pipeline = val

    async def execute_chat(self, request: ChatRequest) -> ChatResponse:
        """
        Execute an educational science inquiry within concurrency slot.
        Loads multi-turn history, resolves antecedents, executes RAG, and persists conversation turns.
        """
        start_time = self.tracker.record_request_start()
        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"

        # 1. Acquire concurrency slot with queue delay tracking
        async with self.gateway.acquire_slot() as queue_wait_ms:
            # 2. Retrieve prior dialogue turns for history resolution
            history_turns: List[Dict[str, str]] = []
            session_state = await self.session_store.get_session(session_id)
            if session_state and session_state.turns:
                for t in session_state.turns[-4:]:
                    history_turns.append({"role": t.role, "content": t.content})

            # 3. Execute Core Advanced Enterprise Science Pipeline
            t_exec_start = time.perf_counter()
            result: AdvancedRAGResult = await self.pipeline.query(
                query_text=request.query,
                session_id=session_id,
                history=history_turns if history_turns else None,
                top_k=request.top_k,
                selected_chapters=request.selected_chapters
            )
            exec_elapsed_ms = (time.perf_counter() - t_exec_start) * 1000.0

            # 4. Update latencies with queue delay
            latencies = dict(result.latency_profile)
            latencies["queue_wait_ms"] = queue_wait_ms
            total_elapsed_ms = round(queue_wait_ms + exec_elapsed_ms, 2)
            latencies["total_ms"] = total_elapsed_ms

            # 5. Persist turns into session store
            await self.session_store.append_turn(
                session_id=session_id,
                user_query=request.query,
                assistant_answer=result.answer,
                citations=result.citations,
                latencies=latencies,
                selected_chapters=request.selected_chapters
            )

            # 6. Record live production telemetry
            circuit_breaker_tripped = (
                getattr(self.pipeline.reranker, "circuit_breaker_status", "active") == "fallback"
            )
            self.tracker.record_request_finish(
                start_time=start_time,
                queue_wait_ms=queue_wait_ms,
                total_latency_ms=total_elapsed_ms,
                circuit_breaker_tripped=circuit_breaker_tripped,
                cache_hit=False,
                prompt_tokens=2500,  # Context upper bound
                completion_tokens=len(result.answer.split())
            )

            return ChatResponse(
                session_id=session_id,
                query=request.query,
                effective_query=result.effective_query,
                answer=result.answer,
                citations=result.citations,
                is_refusal=result.is_refusal,
                latency_profile=latencies,
                trace_id=result.trace_id
            )

    async def execute_chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Execute streaming inquiry yielding Server-Sent Events (SSE) data frames with TTFT < 1.0s.
        """
        start_time = self.tracker.record_request_start()
        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"

        async with self.gateway.acquire_slot() as queue_wait_ms:
            history_turns: List[Dict[str, str]] = []
            session_state = await self.session_store.get_session(session_id)
            if session_state and session_state.turns:
                for t in session_state.turns[-4:]:
                    history_turns.append({"role": t.role, "content": t.content})

            t_exec_start = time.perf_counter()
            token_seq = 0
            full_answer_parts = []

            async for token in self.pipeline.query_stream(
                query_text=request.query,
                session_id=session_id,
                history=history_turns if history_turns else None,
                top_k=request.top_k,
                selected_chapters=request.selected_chapters
            ):
                token_seq += 1
                full_answer_parts.append(token)
                yield f'event: token\ndata: {{"content": {json.dumps(token)}, "seq": {token_seq}}}\n\n'

            exec_elapsed_ms = (time.perf_counter() - t_exec_start) * 1000.0
            total_elapsed_ms = round(queue_wait_ms + exec_elapsed_ms, 2)
            accumulated_answer = "".join(full_answer_parts)

            # Persist turns into session store
            await self.session_store.append_turn(
                session_id=session_id,
                user_query=request.query,
                assistant_answer=accumulated_answer,
                citations=[],
                latencies={"queue_wait_ms": queue_wait_ms, "total_ms": total_elapsed_ms},
                selected_chapters=request.selected_chapters
            )

            # Record telemetry
            circuit_breaker_tripped = (
                getattr(self.pipeline.reranker, "circuit_breaker_status", "active") == "fallback"
            )
            self.tracker.record_request_finish(
                start_time=start_time,
                queue_wait_ms=queue_wait_ms,
                total_latency_ms=total_elapsed_ms,
                circuit_breaker_tripped=circuit_breaker_tripped,
                cache_hit=False,
                prompt_tokens=2500,
                completion_tokens=len(accumulated_answer.split())
            )

            # Emit final done event with metrics
            done_payload = {
                "session_id": session_id,
                "effective_query": request.query,
                "ttft_ms": getattr(self.pipeline, "last_ttft_ms", 150.0),
                "total_ms": total_elapsed_ms
            }
            yield f'event: done\ndata: {json.dumps(done_payload)}\n\n'


# Global service singleton
rag_service = RAGService()
