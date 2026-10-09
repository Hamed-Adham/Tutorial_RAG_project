#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-a-Judge execution wrapper using the 5-Part Standard Rubrics.
Sends evaluation prompts to the local LLM endpoint and parses structured verdicts.
"""

import json
import re
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
from dataclasses import dataclass

from rag_core.config import settings
from rag_core.evaluation.rubrics import get_rubric, JudgeRubric

@dataclass
class JudgeVerdict:
    metric_name: str
    verdict: int  # 1 for Pass, 0 for Fail
    thought: str
    raw_response: str
    latency_seconds: float = 0.0
    error: Optional[str] = None

class LLMJudge:
    """Executes 5-part rubric evaluations against local LLM endpoint."""

    def __init__(
        self,
        base_url: str = settings.nine_router_base_url,
        api_key: str = settings.nine_router_api_key,
        model: str = settings.llm_model,
        timeout: int = 90
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def evaluate_metric(
        self,
        rubric_name: str,
        query: str,
        response: str,
        context: Optional[str] = None,
        ground_truth: Optional[str] = None,
        expected_route: Optional[str] = None,
        actual_route: Optional[str] = None
    ) -> JudgeVerdict:
        """Evaluate a single rubric metric with CoT parsing and latency tracking."""
        start_time = time.time()
        rubric = get_rubric(rubric_name)
        prompt = rubric.format_prompt(
            query=query,
            response=response,
            context=context,
            ground_truth=ground_truth,
            expected_route=expected_route,
            actual_route=actual_route
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": rubric.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "stream": False
        }

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        last_error = None
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    raw_text = data["choices"][0]["message"]["content"].strip()
                    elapsed = round(time.time() - start_time, 3)
                    
                    # Extract JSON from response (supports markdown fenced code blocks)
                    json_match = re.search(r"\{[\s\S]*\}", raw_text)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        verdict_val = int(parsed.get("verdict", 0))
                        thought_val = parsed.get("thought", "")
                        print(f"  [LLM-as-a-Judge] Evaluated '{rubric_name}' in {elapsed:.2f}s | Verdict: {verdict_val}")
                        return JudgeVerdict(
                            metric_name=rubric_name,
                            verdict=verdict_val,
                            thought=thought_val,
                            raw_response=raw_text,
                            latency_seconds=elapsed
                        )
                    else:
                        # Fallback verdict extraction
                        verdict_val = 1 if "1" in raw_text else 0
                        print(f"  [LLM-as-a-Judge] Evaluated '{rubric_name}' (JSON fallback) in {elapsed:.2f}s | Verdict: {verdict_val}")
                        return JudgeVerdict(
                            metric_name=rubric_name,
                            verdict=verdict_val,
                            thought="Could not parse strict JSON object; extracted fallback.",
                            raw_response=raw_text,
                            latency_seconds=elapsed,
                            error="JSON parse failed"
                        )

            except Exception as e:
                last_error = e
                print(f"  [Judge warning] Attempt {attempt}/3 failed ({e}). Retrying in 2s...")
                time.sleep(2)

        elapsed = round(time.time() - start_time, 3)
        print(f"  [LLM-as-a-Judge] Failed '{rubric_name}' after {elapsed:.2f}s | Error: {last_error}")
        return JudgeVerdict(
            metric_name=rubric_name,
            verdict=0,
            thought="Evaluation call failed after 3 attempts.",
            raw_response="",
            latency_seconds=elapsed,
            error=str(last_error)
        )
