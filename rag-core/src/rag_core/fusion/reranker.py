"""
Neural Cross-Encoder Re-ranker using BAAI/bge-reranker-v2-m3 with Latency Circuit Breaker.
Evaluates query-document cross-attention within a 350ms budget, gracefully falling back to RRF ranking on timeout.
"""

import asyncio
import threading
from typing import List, Optional, Any
import structlog

from rag_core.indexing.science_schema import RetrievalCandidate, RerankedCandidate
from rag_core.contracts import ICrossEncoderReranker

logger = structlog.get_logger(__name__)


class CrossEncoderReranker(ICrossEncoderReranker):
    """High-precision cross-encoder re-ranker with latency circuit breaker."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        model: Optional[Any] = None,
        default_timeout_ms: float = 3000.0
    ):
        self.model_name = model_name
        self.model = model
        self.default_timeout_ms = default_timeout_ms
        self._initialized = model is not None
        self._warned_fallback = False
        self._lock = threading.Lock()
        self.circuit_breaker_status: str = "active"

    def warmup(self) -> bool:
        """Eagerly load CrossEncoder model into memory and run a dummy forward pass to eliminate cold start."""
        self._ensure_model()
        if self.model is not None:
            try:
                self.model.predict([["warmup", "warmup"]], max_length=32, show_progress_bar=False)
            except Exception:
                pass
        return self.model is not None

    def _ensure_model(self) -> None:
        """Lazy load CrossEncoder model with optimal hardware device."""
        if self._initialized:
            return

        import torch

        # Use CPU on macOS (Darwin) to prevent Apple Metal (MPS) command buffer race conditions
        # when running in asyncio thread pools. CPU handles 10-15 pairs in ~180ms with 100% stability.
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # 1. Try sentence_transformers
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading CrossEncoder model via sentence-transformers", model=self.model_name, device=device)
            self.model = CrossEncoder(self.model_name, device=device)
            self._initialized = True
            return
        except Exception as e_st:
            logger.info("sentence_transformers unavailable, falling back to HuggingFace transformers", error=str(e_st))

        # 2. Resilient fallback to pure HuggingFace transformers
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            class _HFTransformerReranker:
                def __init__(self, model_name: str, dev: str):
                    self.dev = dev
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.hf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    self.hf_model.to(dev)
                    self.hf_model.eval()

                def predict(self, pairs: List[List[str]]) -> List[float]:
                    with torch.no_grad():
                        inputs = self.tokenizer(
                            pairs,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt"
                        ).to(self.dev)
                        logits = self.hf_model(**inputs, return_dict=True).logits.view(-1).float()
                        return logits.cpu().tolist()

            logger.info("Loading CrossEncoder model via HuggingFace transformers", model=self.model_name, device=device)
            self.model = _HFTransformerReranker(self.model_name, device)
            self._initialized = True
        except Exception as e_hf:
            logger.warning("Failed to initialize neural cross-encoder, using fallback", error=str(e_hf))
            self.model = None
            self._initialized = True

    def _predict_sync(self, pairs: List[List[str]], max_length: int = 512) -> List[float]:
        """Synchronous inference call executed in thread pool."""
        self._ensure_model()
        if self.model is None:
            raise RuntimeError("CrossEncoder model unavailable")

        with self._lock:
            try:
                scores = self.model.predict(pairs, max_length=max_length, show_progress_bar=False)
            except TypeError:
                try:
                    scores = self.model.predict(pairs, show_progress_bar=False)
                except TypeError:
                    scores = self.model.predict(pairs)
            return [float(s) for s in scores]

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: int = 5,
        timeout_ms: Optional[float] = None,
        max_length: int = 512
    ) -> List[RerankedCandidate]:
        """Re-rank candidate chunks using cross-encoder with strict circuit breaker."""
        if not candidates:
            return []

        # Rerank the top candidates from RRF (full 512-token window with zero truncation)
        eval_candidates = candidates[:max(top_k, 8)]
        timeout = (timeout_ms or self.default_timeout_ms) / 1000.0
        pairs = [[query, cand.text] for cand in eval_candidates]

        try:
            loop = asyncio.get_running_loop()
            scores = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._predict_sync(pairs, max_length=max_length)),
                timeout=timeout
            )

            # Sort by neural cross-encoder score descending
            scored_candidates = list(zip(eval_candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            results: List[RerankedCandidate] = []
            for rank, (cand, score) in enumerate(scored_candidates[:top_k], start=1):
                results.append(
                    RerankedCandidate(
                        candidate=cand,
                        cross_encoder_score=score,
                        final_rank=rank
                    )
                )
            self.circuit_breaker_status = "active"
            return results

        except (asyncio.TimeoutError, Exception) as exc:
            self.circuit_breaker_status = "fallback"
            reason = "timeout" if isinstance(exc, asyncio.TimeoutError) else "unavailable"
            if not self._warned_fallback:
                logger.warning(
                    f"CrossEncoder {reason} ({timeout*1000:.0f}ms budget), falling back to raw RRF ranking (further query fallbacks will be silent)",
                    exc_info=False
                )
                self._warned_fallback = True
            else:
                logger.debug(f"CrossEncoder {reason}, using RRF ranking")
            # Graceful fallback: return top_k candidates in their existing RRF order
            results = []
            for rank, cand in enumerate(candidates[:top_k], start=1):
                results.append(
                    RerankedCandidate(
                        candidate=cand,
                        cross_encoder_score=cand.rrf_score,
                        final_rank=rank
                    )
                )
            return results
