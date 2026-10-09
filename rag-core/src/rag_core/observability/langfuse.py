#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Langfuse OpenTelemetry Tracing Module.
Provides centralized trace, span, and score management across all RAG pipeline stages,
adhering to OpenInference standards, Langfuse SDK v4, and Constitution v1.4.0.
"""

import time
from typing import Dict, Any, List, Optional
from rag_core.config import settings

try:
    from langfuse import Langfuse, propagate_attributes
except ImportError:
    Langfuse = None
    propagate_attributes = None


class DummySpan:
    """Safe fallback span object when Langfuse is offline or disabled."""
    def __init__(self, *args, **kwargs):
        pass

    def span(self, *args, **kwargs):
        return DummySpan()

    def start_observation(self, *args, **kwargs):
        return DummySpan()

    def generation(self, *args, **kwargs):
        return DummySpan()

    def event(self, *args, **kwargs):
        return DummySpan()

    def score(self, *args, **kwargs):
        return DummySpan()

    def score_trace(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass


class DummyTrace(DummySpan):
    """Safe fallback trace object."""
    id: str = "offline_trace"
    trace_id: str = "offline_trace"


class LangfuseTracker:
    """
    Centralized Langfuse telemetry manager adhering to OpenInference standards,
    Langfuse SDK v4, and Constitution v1.4.0 with graceful offline fallback.
    """

    def __init__(self, release: str = "v1.4.0"):
        self.release = release
        self.client: Optional[Any] = None
        if settings.langfuse_enabled and Langfuse is not None:
            try:
                self.client = Langfuse(
                    host=settings.langfuse_host,
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key
                )
            except Exception as e:
                # Log once, fall back gracefully
                print(f"Notice: Langfuse client initialization deferred/offline: {e}")

    @property
    def is_active(self) -> bool:
        """Return True if Langfuse client is active and enabled."""
        return self.client is not None

    def get_langchain_handler(
        self,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Return native Langfuse CallbackHandler for LangChain / LangGraph execution.
        """
        if not self.client:
            return None
        try:
            from langfuse.langchain import CallbackHandler
            handler = CallbackHandler(
                public_key=settings.langfuse_public_key,
            )
            return handler
        except Exception:
            return None

    def sync_dataset(
        self,
        dataset_name: str,
        items: List[Dict[str, Any]],
        description: str = "Unified RAG Golden Benchmark Dataset"
    ) -> int:
        """
        Upload or synchronize benchmark items to a persistent Langfuse Dataset.
        Returns count of synced items.
        """
        if not self.client:
            return 0
        try:
            try:
                self.client.create_dataset(name=dataset_name, description=description)
            except Exception:
                pass

            count = 0
            for it in items:
                input_payload = {
                    "query": it.get("query"),
                    "suite": it.get("suite"),
                    "difficulty": it.get("difficulty")
                }
                expected_output = {
                    "ground_truth": it.get("ground_truth"),
                    "expected_refusal": it.get("expected_refusal", False),
                    "expected_route": it.get("expected_route")
                }
                self.client.create_dataset_item(
                    dataset_name=dataset_name,
                    input=input_payload,
                    expected_output=expected_output,
                    metadata={"source_document": it.get("source_document")}
                )
                count += 1
            return count
        except Exception as e:
            print(f"Notice: Failed to sync dataset to Langfuse: {e}")
            return 0

    def create_trace(
        self,
        name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_data: Optional[Any] = None,
    ) -> Any:
        """Create and return a root trace object."""
        if not self.client:
            return DummyTrace()
        try:
            combined_tags = tags or []
            combined_metadata = metadata or {}
            if propagate_attributes:
                with propagate_attributes(
                    tags=combined_tags,
                    session_id=session_id,
                    user_id=user_id,
                    trace_name=name,
                    version=self.release,
                    metadata=combined_metadata,
                ):
                    return self.client.start_observation(
                        name=name,
                        as_type="chain",
                        input=input_data,
                        metadata=combined_metadata,
                        version=self.release,
                    )
            return self.client.start_observation(
                name=name,
                as_type="chain",
                input=input_data,
                metadata=combined_metadata,
                version=self.release,
            )
        except Exception:
            return DummyTrace()

    def trace_query(
        self,
        name: str,
        question: str,
        answer: str,
        top_k: int,
        retrieved_count: int,
        sources: List[Dict[str, Any]],
        retrieval_latency: float,
        generation_latency: float,
        model_name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Create a structured parent trace with nested retrieval (as_type='retriever')
        and synthesis (as_type='generation') observations adhering to Langfuse v4 standards.
        Returns the trace ID for downstream score ingestion.
        """
        if not self.client:
            return "offline_trace"

        try:
            combined_tags = ["pipeline:rag"]
            if tags:
                combined_tags.extend(tags)
            combined_metadata = metadata or {}

            def _build_observations():
                root = self.client.start_observation(
                    name=name,
                    as_type="chain",
                    input={"question": question, "top_k": top_k},
                    output={"answer": answer},
                    metadata=combined_metadata,
                    version=self.release,
                )

                # Retrieval span (Observation type: RETRIEVER)
                retrieval_span = root.start_observation(
                    name="retrieval",
                    as_type="retriever",
                    input={"question": question, "top_k": top_k},
                    output={"retrieved_count": retrieved_count, "sources": sources},
                    metadata={
                        "latency_seconds": round(retrieval_latency, 3),
                        "vector_db": "qdrant",
                    },
                )
                retrieval_span.end()

                # Generation span (Observation type: GENERATION)
                gen_span = root.start_observation(
                    name="synthesis",
                    as_type="generation",
                    input=question,
                    output=answer,
                    model=model_name or settings.llm_model,
                    usage_details=usage_details,
                    cost_details=cost_details,
                    metadata={"latency_seconds": round(generation_latency, 3)},
                )
                gen_span.end()

                root.end()
                return getattr(root, "trace_id", "offline_trace")

            if propagate_attributes:
                with propagate_attributes(
                    tags=combined_tags,
                    session_id=session_id,
                    user_id=user_id,
                    trace_name=name,
                    version=self.release,
                    metadata=combined_metadata,
                ):
                    return _build_observations()
            else:
                return _build_observations()

        except Exception as e:
            return "offline_trace"

    def add_trace_score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record an empirical evaluation score linked to a trace ID in Langfuse.
        Supported metric names: gate_pass, faithfulness, answer_relevancy, context_precision, context_recall, judge_latency.
        """
        if not self.client or not trace_id or trace_id == "offline_trace":
            return

        try:
            # Langfuse v4 API: create_score
            self.client.create_score(
                trace_id=trace_id,
                name=name,
                value=float(value),
                comment=comment,
                metadata=metadata,
            )
        except Exception:
            try:
                if hasattr(self.client, "score"):
                    self.client.score(
                        trace_id=trace_id,
                        name=name,
                        value=float(value),
                        comment=comment,
                    )
            except Exception:
                pass

    def flush(self):
        """Flush queued telemetry events to Langfuse backend."""
        if self.client:
            try:
                self.client.flush()
            except Exception:
                pass

    def shutdown(self):
        """Gracefully shutdown telemetry client."""
        if self.client:
            try:
                if hasattr(self.client, "shutdown"):
                    self.client.shutdown()
                else:
                    self.client.flush()
            except Exception:
                pass
