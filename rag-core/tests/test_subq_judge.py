#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for SubQJudge (Feature 007).
Validates System One rubrics, parallel batching, and graceful fallback to local LLMJudge.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from rag_core.evaluation.subq_judge import SubQJudge
from rag_core.evaluation.benchmark_schema import SubQJudgeVerdict
from rag_core.evaluation.judge_evaluator import JudgeVerdict


@pytest.mark.asyncio
async def test_subq_judge_faithfulness_success():
    """Verify SubQ judge parses successful faithfulness response."""
    judge = SubQJudge(api_key="valid_test_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "answers": {
            "faithfulness": {
                "choice": "faithful",
                "confidence": 0.95,
                "score": 1.0,
            }
        },
        "usage": {"total_tokens": 120},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        verdict = await judge.evaluate_faithfulness(
            query="قلب چند حفره دارد؟",
            context="قلب از چهار حفره شامل دو دهلیز و دو بطن تشکیل شده است.",
            response="قلب دارای چهار حفره است.",
        )

        assert verdict.verdict == 1
        assert verdict.metric_name == "faithfulness"
        assert verdict.confidence == 0.95
        assert verdict.fallback_used is False


@pytest.mark.asyncio
async def test_subq_judge_hallucination_detected():
    """Verify SubQ judge detects ungrounded claims."""
    judge = SubQJudge(api_key="valid_test_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "answers": {
            "faithfulness": {
                "choice": "hallucinated",
                "confidence": 0.91,
                "score": 0.0,
            }
        },
        "usage": {"total_tokens": 150},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        verdict = await judge.evaluate_faithfulness(
            query="قلب چند حفره دارد؟",
            context="قلب از چهار حفره تشکیل شده است.",
            response="قلب از هشت حفره ساخته شده و دارای بالشتک است.",
        )

        assert verdict.verdict == 0
        assert verdict.score == 0.0


@pytest.mark.asyncio
async def test_subq_judge_fallback_on_network_error():
    """Verify graceful fallback to local LLMJudge when SubQ API fails."""
    mock_llm = MagicMock()
    mock_llm.evaluate_metric.return_value = JudgeVerdict(
        metric_name="faithfulness",
        verdict=1,
        thought="Fallback local judge verified groundedness.",
        raw_response='{"verdict": 1}',
        latency_seconds=0.45,
    )

    judge = SubQJudge(api_key="valid_key", llm_fallback=mock_llm)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = Exception("Connection refused")
        verdict = await judge.evaluate_faithfulness(
            query="تست",
            context="متن آزمایشی",
            response="پاسخ آزمایشی",
        )

        assert verdict.verdict == 1
        assert verdict.fallback_used is True
        assert "Fallback" in verdict.thought


@pytest.mark.asyncio
async def test_subq_judge_batch_parallel():
    """Verify parallel batch evaluation across multiple items."""
    judge = SubQJudge(api_key="dummy_key")

    # Mock the evaluate_faithfulness and evaluate_relevancy directly to test batching logic
    mock_v = SubQJudgeVerdict(
        metric_name="faithfulness",
        verdict=1,
        score=1.0,
        confidence=0.9,
        latency_seconds=0.05,
    )
    judge.evaluate_faithfulness = AsyncMock(return_value=mock_v)
    judge.evaluate_relevancy = AsyncMock(return_value=mock_v)

    items = [{"query": f"Q{i}", "answer": f"A{i}", "context": f"C{i}"} for i in range(5)]
    results = await judge.evaluate_batch_parallel(items, batch_size=2)

    assert len(results) == 5
    for r in results:
        assert "faithfulness" in r
        assert "relevancy" in r
