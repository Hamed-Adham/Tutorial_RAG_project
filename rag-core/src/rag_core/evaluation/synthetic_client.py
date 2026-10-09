#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Evaluation Dataset Generation Client.
Connects to local Nine Router / OpenAI-compatible gateways to synthesize diverse,
colloquial, multi-hop, and hard-negative Persian evaluation queries.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
import httpx

from rag_core.config import settings

logger = logging.getLogger(__name__)


class SyntheticGenerationClient:
    """
    HTTP client for batch-synthesizing evaluation items via Nine Router / OpenAI endpoints.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3
    ):
        self.base_url = (base_url or settings.nine_router_base_url).rstrip("/")
        self.api_key = api_key or settings.nine_router_api_key
        self.model = model or settings.benchmark_model
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(
        self,
        prompt: str,
        system_instruction: str = "You are an expert Persian educational and technical data benchmark designer.",
        temperature: float = 0.7,
        json_mode: bool = True
    ) -> str:
        """
        Execute completion with retries and exponential backoff.
        """
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(endpoint, headers=headers, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        choices = data.get("choices", [])
                        if choices:
                            return choices[0].get("message", {}).get("content", "").strip()
                        raise ValueError(f"No choices in LLM response: {data}")
                    else:
                        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
            except Exception as e:
                last_error = e
                logger.warning(f"Synthetic generation attempt {attempt} failed: {e}")
                time.sleep(1.5 ** attempt)

        raise RuntimeError(f"Synthetic generation failed after {self.max_retries} retries: {last_error}")

    def complete_json(
        self,
        prompt: str,
        system_instruction: str = "You are an expert Persian data benchmark designer. Emit valid JSON only.",
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Executes completion and parses JSON output safely.
        """
        raw_text = self.complete(prompt, system_instruction, temperature=temperature, json_mode=True)
        # Strip potential markdown code fences
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
