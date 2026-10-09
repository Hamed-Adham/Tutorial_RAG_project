#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TypeSafe Jev System One Client.
Provides resilient, typed interaction with https://api.typesafe.ai/v1/systemone
with automatic retry handling and error reporting.
"""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

from rag_core.config import settings


class TypeSafeJevClient:
    """
    Lightweight, dependency-free HTTP client for TypeSafe's Jev System One API.
    Handles authentication, payload formatting, retries, and network error detection.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2
    ):
        # Primary source is the local Nine Router gateway (see config.py); the
        # legacy TypeSafe.ai keys in .env remain available as a fallback.
        self.api_key = api_key or settings.resolved_jev_api_key
        self.base_url = base_url or settings.jev_base_url
        self.model = model or settings.jev_model
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def is_configured(self) -> bool:
        """Return True if an API key is present."""
        return bool(self.api_key and self.api_key.strip())

    def system_one(
        self,
        state: Any,
        questions: Dict[str, Dict[str, Any]],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a System One evaluation request against TypeSafe Jev.

        Args:
            state: Context or application state (string, dict, or list).
            questions: Map of question IDs to question specifications.
            model: Optional model override (defaults to self.model).

        Returns:
            Dictionary matching TypeSafe API response (containing "answers", "usage", etc.).

        Raises:
            RuntimeError: If unconfigured, unauthorized, or connection fails.
        """
        if not self.is_configured:
            raise RuntimeError("TypeSafe JEV_API_KEY is not configured in environment or settings.")

        payload = {
            "state": state,
            "model": model or self.model,
            "questions": questions
        }
        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "UnifiedRAG-TypeSafe/1.0"
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.base_url, data=data_bytes, headers=headers, method="POST")
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw_body = resp.read().decode("utf-8")
                    parsed = json.loads(raw_body)
                    parsed["_latency_seconds"] = time.perf_counter() - t0
                    return parsed
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                last_error = RuntimeError(f"TypeSafe API HTTP {e.code}: {err_body}")
                # Don't retry client errors (400, 401, 403)
                if 400 <= e.code < 500:
                    raise last_error
            except Exception as e:
                last_error = RuntimeError(f"TypeSafe connection failed: {e}")

            if attempt < self.max_retries:
                time.sleep(0.5 * (attempt + 1))

        raise last_error
