#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SubQ System One LLM-as-a-Judge (Feature 007).
Executes high-throughput (< 600ms), typed System One rubric evaluations
using the SubQ Decisions API (subq-snap-preview, 384K context window)
with automatic fallback to local generative LLMJudge (Gemini via Nine Router).
"""

import time
import asyncio
import httpx
from typing import Dict, Any, List, Optional
import structlog

from rag_core.config import settings
from rag_core.evaluation.benchmark_schema import SubQJudgeVerdict
from rag_core.evaluation.judge_evaluator import LLMJudge

logger = structlog.get_logger(__name__)


class SubQJudge:
    """
    High-throughput System One Evaluator powered by SubQ Decisions API (subq-snap-preview).
    Evaluates Faithfulness, Relevancy, Citation Attribution, and Turn Relevance.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        llm_fallback: Optional[LLMJudge] = None,
        timeout: float = 20.0,
        max_retries: int = 2,
    ):
        self.endpoint = endpoint or getattr(settings, "subq_endpoint", "https://api.subq.ai/v1/systemone")
        self.api_key = api_key or getattr(settings, "subq_api_key", None)
        self.model = model or getattr(settings, "subq_model", "subq-snap-preview")
        self.llm_fallback = llm_fallback or LLMJudge()
        self.timeout = timeout
        self.max_retries = max_retries

    async def evaluate_faithfulness(
        self,
        query: str,
        context: str,
        response: str,
    ) -> SubQJudgeVerdict:
        """
        Evaluate factual groundedness (preventing hallucination).
        Returns verdict 1 (Faithful) or 0 (Hallucinated).
        """
        questions = {
            "faithfulness": {
                "type": "choice",
                "instructions": (
                    "Evaluate if all claims, facts, numbers, and scientific statements in `agent_response` "
                    "are directly supported by and faithful to `retrieved_context`. "
                    "Select 'faithful' only if no outside ungrounded facts are introduced."
                ),
                "criteria": {
                    "faithful": "Every claim in response is strictly affirmed by the context without extrapolation.",
                    "hallucinated": "Contains ungrounded claims, altered facts, or outside information not in context.",
                },
            }
        }
        state = {
            "user_query": query,
            "retrieved_context": context,
            "agent_response": response,
        }

        verdict_dict = await self._evaluate_single_with_fallback(
            state=state,
            questions=questions,
            target_metric="faithfulness",
            fallback_rubric="faithfulness",
            query=query,
            response=response,
            context=context,
        )
        return verdict_dict

    async def evaluate_relevancy(
        self,
        query: str,
        response: str,
    ) -> SubQJudgeVerdict:
        """
        Evaluate answer pertinence to the user query.
        Returns verdict 1 (Relevant) or 0 (Irrelevant/Evasive).
        """
        questions = {
            "relevancy": {
                "type": "choice",
                "instructions": (
                    "Evaluate if `agent_response` directly and accurately answers `user_query` "
                    "without evading the core inquiry or introducing unrelated tangents."
                ),
                "criteria": {
                    "relevant": "Directly and fully addresses the question asked by the user.",
                    "irrelevant": "Evades the question, answers an unrelated topic, or gives an incomplete non-answer.",
                },
            }
        }
        state = {
            "user_query": query,
            "agent_response": response,
        }

        verdict_dict = await self._evaluate_single_with_fallback(
            state=state,
            questions=questions,
            target_metric="relevancy",
            fallback_rubric="relevancy",
            query=query,
            response=response,
        )
        return verdict_dict

    async def evaluate_citation_attribution(
        self,
        claim: str,
        citation_source: str,
        context: str,
    ) -> SubQJudgeVerdict:
        """
        Evaluate if a cited source genuinely contains and supports the attributed claim.
        """
        questions = {
            "citation_attribution": {
                "type": "choice",
                "instructions": (
                    f"Does the reference context from '{citation_source}' specifically contain and support "
                    f"the claim made in: '{claim}'?"
                ),
                "criteria": {
                    "supported": "The cited context explicitly supports the claim made.",
                    "ghost_citation": "The cited context does not support the claim or mentions something else.",
                },
            }
        }
        state = {
            "claim": claim,
            "citation_source": citation_source,
            "retrieved_context": context,
            "agent_response": claim,
            "user_query": f"Verify citation for: {claim}",
        }

        verdict_dict = await self._evaluate_single_with_fallback(
            state=state,
            questions=questions,
            target_metric="citation_attribution",
            fallback_rubric="relevancy",
            query=claim,
            response=claim,
            context=context,
        )
        return verdict_dict

    async def _evaluate_single_with_fallback(
        self,
        state: Dict[str, Any],
        questions: Dict[str, Any],
        target_metric: str,
        fallback_rubric: str,
        query: str,
        response: str,
        context: Optional[str] = None,
    ) -> SubQJudgeVerdict:
        """Execute SubQ call with automatic local LLMJudge fallback."""
        t0 = time.perf_counter()

        # If API key is missing or dummy, trigger fallback directly
        if not self.api_key or "your_" in self.api_key:
            return self._execute_fallback(
                target_metric=target_metric,
                fallback_rubric=fallback_rubric,
                query=query,
                response=response,
                context=context,
                reason="SubQ API key not configured",
            )

        payload = {
            "model": self.model,
            "state": state,
            "questions": questions,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "UnifiedRAG-SubQJudge/1.0",
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(self.endpoint, json=payload, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        elapsed = round(time.perf_counter() - t0, 3)
                        return self._parse_subq_verdict(data, target_metric, elapsed)
                    elif 400 <= resp.status_code < 500:
                        logger.warning("SubQ client error; falling back to local judge", status_code=resp.status_code)
                        break
                    else:
                        resp.raise_for_status()
            except Exception as e:
                last_error = e
                await asyncio.sleep(0.1 * attempt)

        # Fallback to local LLMJudge
        return self._execute_fallback(
            target_metric=target_metric,
            fallback_rubric=fallback_rubric,
            query=query,
            response=response,
            context=context,
            reason=f"SubQ request failed ({last_error})",
        )

    def _parse_subq_verdict(
        self,
        data: Dict[str, Any],
        target_metric: str,
        elapsed: float
    ) -> SubQJudgeVerdict:
        """Parse structured SubQ response into typed SubQJudgeVerdict."""
        answers = data.get("answers") or data.get("decisions") or {}
        ans = answers.get(target_metric, {})
        choice = ""
        confidence = 1.0
        score = 1.0

        if isinstance(ans, dict):
            choice = str(ans.get("choice") or ans.get("value") or "").strip().lower()
            confidence = float(ans.get("confidence", 1.0))
            score = float(ans.get("score", 1.0 if choice in ("faithful", "relevant", "supported") else 0.0))
        elif isinstance(ans, str):
            choice = ans.strip().lower()

        is_pass = 1 if choice in ("faithful", "relevant", "supported", "yes", "true") else 0
        tokens = data.get("usage", {}).get("total_tokens", 0)

        return SubQJudgeVerdict(
            metric_name=target_metric,
            verdict=is_pass,
            score=score,
            confidence=confidence,
            thought=f"SubQ choice: '{choice}' with confidence {confidence:.2f}",
            raw_response=data,
            latency_seconds=elapsed,
            fallback_used=False,
            tokens_used=tokens,
        )

    def _execute_fallback(
        self,
        target_metric: str,
        fallback_rubric: str,
        query: str,
        response: str,
        context: Optional[str] = None,
        reason: str = "Fallback triggered",
    ) -> SubQJudgeVerdict:
        """Run local LLMJudge synchronously in thread executor as fallback."""
        t0 = time.perf_counter()
        try:
            verdict = self.llm_fallback.evaluate_metric(
                rubric_name=fallback_rubric,
                query=query,
                response=response,
                context=context,
            )
            elapsed = round(time.perf_counter() - t0, 3)
            return SubQJudgeVerdict(
                metric_name=target_metric,
                verdict=verdict.verdict,
                score=float(verdict.verdict),
                confidence=0.85,
                thought=f"Fallback LLMJudge used ({reason}): {verdict.thought}",
                raw_response={"fallback": True, "raw": verdict.raw_response},
                latency_seconds=elapsed,
                fallback_used=True,
                tokens_used=0,
            )
        except Exception as e:
            elapsed = round(time.perf_counter() - t0, 3)
            return SubQJudgeVerdict(
                metric_name=target_metric,
                verdict=0,
                score=0.0,
                confidence=0.0,
                thought=f"Fallback LLMJudge also failed: {e}",
                raw_response={"error": str(e)},
                latency_seconds=elapsed,
                fallback_used=True,
                tokens_used=0,
            )

    async def evaluate_batch_parallel(
        self,
        items: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> List[Dict[str, SubQJudgeVerdict]]:
        """
        Evaluate multiple items concurrently with 10-item chunking.
        """
        results: List[Dict[str, SubQJudgeVerdict]] = []

        async def _eval_item(it: Dict[str, Any]) -> Dict[str, SubQJudgeVerdict]:
            q = it.get("query") or it.get("input", "")
            ans = it.get("answer") or it.get("response", "")
            ctx = it.get("context", "")

            # Concurrently evaluate faithfulness and relevancy
            faith_task = self.evaluate_faithfulness(query=q, context=ctx, response=ans)
            rel_task = self.evaluate_relevancy(query=q, response=ans)
            f_res, r_res = await asyncio.gather(faith_task, rel_task)
            return {"faithfulness": f_res, "relevancy": r_res}

        # Process in batches
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            chunk_results = await asyncio.gather(*[_eval_item(it) for it in chunk])
            results.extend(chunk_results)

        return results
