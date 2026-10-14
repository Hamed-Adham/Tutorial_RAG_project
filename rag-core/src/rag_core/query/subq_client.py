# -*- coding: utf-8 -*-
"""
SubQ Decision Client for System One Decision Model 2.
Evaluates historical conversational turns against the current query using:
- Endpoint: POST https://api.subq.ai/v1/systemone
- Model: subq-snap-preview (384K context window)
- Batching: Concurrent 10-10 chunking via asyncio.gather to respect SubQ's 16-question limit.
"""

import asyncio
import re
from typing import Dict, Any, List, Optional
import httpx
import structlog

from rag_core.config import settings
from rag_core.indexing.science_schema import HistoryRelevanceTier

logger = structlog.get_logger(__name__)


class SubQDecisionClient:
    """
    Asynchronous client for SubQ Decisions API (System One Model 2).
    Evaluates up to 20 candidate turns using parallel 10-10 chunking.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 3.0,
        max_retries: int = 1
    ):
        self.api_key = api_key or settings.subq_api_key
        self.endpoint = endpoint or settings.subq_endpoint
        self.model = model or settings.subq_model
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def is_configured(self) -> bool:
        """Return True if API key is present."""
        return bool(self.api_key and self.api_key.strip())

    async def evaluate_history_turns(
        self,
        query: str,
        candidate_turns: List[Dict[str, Any]]
    ) -> Dict[str, HistoryRelevanceTier]:
        """
        Evaluate candidate conversational turns against current query.
        Splits candidate turns into batches of <= 10 questions to respect the 16-question limit,
        executing them concurrently via asyncio.gather.

        Args:
            query: Current user inquiry.
            candidate_turns: List of historical Q&A dictionaries (up to 20).
                Each item should contain 'turn_id', 'question', 'answer', and optional 'summary'/'q_and_a_summary'.

        Returns:
            Dictionary mapping turn_id to HistoryRelevanceTier.
        """
        if not candidate_turns:
            return {}

        if not self.is_configured:
            logger.warning("SubQ API key is not configured; defaulting all candidate turns to NOT_NEEDED_NOT_RELEVANT")
            return {
                str(t.get("turn_id", f"turn_{i}")): HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT
                for i, t in enumerate(candidate_turns)
            }

        # Chunk candidate turns into batches of at most 10 items (e.g. 10-10 for 20 items)
        chunk_size = 10
        chunks = [candidate_turns[i:i + chunk_size] for i in range(0, len(candidate_turns), chunk_size)]

        tasks = [self._evaluate_batch(query, chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged_evaluations: Dict[str, HistoryRelevanceTier] = {}
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error(
                    "SubQ batch evaluation failed",
                    batch_index=idx,
                    error=str(res)
                )
                # Fallback for failed batch: assign default NOT_NEEDED_NOT_RELEVANT
                for t_idx, t in enumerate(chunks[idx]):
                    tid = str(t.get("turn_id", f"turn_{idx * chunk_size + t_idx}"))
                    merged_evaluations[tid] = HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT
            else:
                merged_evaluations.update(res)

        return merged_evaluations

    async def _evaluate_batch(
        self,
        query: str,
        batch: List[Dict[str, Any]]
    ) -> Dict[str, HistoryRelevanceTier]:
        """
        Execute a single batch evaluation against SubQ System One API.
        """
        if not batch:
            return {}

        key_to_turn_id: Dict[str, str] = {}
        questions: Dict[str, Dict[str, Any]] = {}
        turns_state: List[Dict[str, Any]] = []

        for idx, t in enumerate(batch):
            raw_id = str(t.get("turn_id", f"turn_{idx}"))
            # SubQ question keys must be alphanumeric identifiers (with underscores)
            sanitized_key = f"turn_{re.sub(r'[^a-zA-Z0-9_]', '_', raw_id)}"
            # Avoid key collisions within batch
            if sanitized_key in key_to_turn_id:
                sanitized_key = f"{sanitized_key}_{idx}"
            key_to_turn_id[sanitized_key] = raw_id

            question_text = str(t.get("question", "")).strip()
            answer_text = str(t.get("answer", "")).strip()
            summary_text = str(t.get("summary") or t.get("q_and_a_summary") or "").strip()

            turns_state.append({
                "turn_id": raw_id,
                "question": question_text,
                "answer": answer_text[:500] if len(answer_text) > 500 else answer_text,
                "summary": summary_text
            })

            snippet = question_text[:80] + ("..." if len(question_text) > 80 else "")
            questions[sanitized_key] = {
                "type": "choice",
                "instructions": (
                    f"Evaluate if previous turn '{raw_id}' (User asked: '{snippet}') is needed or relevant "
                    f"to answer the student's current query: '{query}'."
                ),
                "criteria": {
                    "not_needed_not_relevant": "The turn has no relation to the current question and is not required context.",
                    "not_needed_relevant": "The turn covers related science topics, but the current question is self-contained and does not require past context.",
                    "needed_not_relevant": "The turn provides antecedent or pronoun resolution essential to understand the current query, even if the topic seems different.",
                    "needed_relevant": "The turn provides essential context, explanations, or continuity required to accurately answer the current query."
                }
            }

        payload = {
            "model": self.model,
            "state": {
                "current_query": query,
                "instructions": "Evaluate whether previous conversational turns are relevant or needed as context to answer the student's current query accurately.",
                "candidate_turns": turns_state
            },
            "questions": questions
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "UnifiedRAG-SubQClient/1.0"
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            last_err = None
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.post(self.endpoint, json=payload, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        return self._parse_subq_answers(data, key_to_turn_id)
                    elif 400 <= resp.status_code < 500:
                        logger.warning(
                            "SubQ API client error; skipping retries",
                            status_code=resp.status_code,
                            body=resp.text
                        )
                        return {tid: HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT for tid in key_to_turn_id.values()}
                    else:
                        resp.raise_for_status()
                except Exception as exc:
                    last_err = exc
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.1 * (attempt + 1))

            logger.warning("SubQ request failed after retries; defaulting batch", error=str(last_err))
            return {tid: HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT for tid in key_to_turn_id.values()}

    def _parse_subq_answers(
        self,
        data: Dict[str, Any],
        key_to_turn_id: Dict[str, str]
    ) -> Dict[str, HistoryRelevanceTier]:
        """
        Extract answers from SubQ response and map them to HistoryRelevanceTier.
        """
        results: Dict[str, HistoryRelevanceTier] = {}
        answers = data.get("answers") or data.get("decisions") or data.get("results") or {}

        valid_tiers = {
            "not_needed_not_relevant": HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT,
            "not_needed_relevant": HistoryRelevanceTier.NOT_NEEDED_RELEVANT,
            "needed_not_relevant": HistoryRelevanceTier.NEEDED_NOT_RELEVANT,
            "needed_relevant": HistoryRelevanceTier.NEEDED_RELEVANT,
        }

        for q_key, turn_id in key_to_turn_id.items():
            ans_val = answers.get(q_key)
            choice_str = ""
            if isinstance(ans_val, dict):
                choice_str = str(ans_val.get("choice") or ans_val.get("value") or "").strip().lower()
            elif isinstance(ans_val, str):
                choice_str = ans_val.strip().lower()

            tier = valid_tiers.get(choice_str, HistoryRelevanceTier.NOT_NEEDED_NOT_RELEVANT)
            results[turn_id] = tier

        return results
