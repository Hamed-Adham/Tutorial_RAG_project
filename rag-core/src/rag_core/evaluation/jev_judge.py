#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JevJudge: Fast, Calibrated System One Evaluator.
Executes Layer 2.5 evaluation using TypeSafe Jev (Faithfulness Noul, 3-level Relevancy Score, Refusal Noul)
with single-request speculative fan-out and automatic fallback to generative LLMJudge.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from rag_core.evaluation.jev_client import TypeSafeJevClient
from rag_core.naive.judge_evaluator import LLMJudge


@dataclass
class JevEvaluationVerdict:
    verdict: int                  # 1 for Pass, 0 for Fail
    score: float                  # Continuous score or probability (0.0 to 1.0 or 0.0 to 2.0)
    metric_name: str
    confidence: float             # TypeSafe certainty score (0.0 to 1.0)
    latency_seconds: float
    fallback_used: bool = False
    details: Optional[Dict[str, Any]] = None


class JevJudge:
    """
    Tier 2.5 System One Evaluator powered by TypeSafe Jev.
    Executes sub-500ms typed evaluations with zero generative token cost.
    Degrades gracefully to local LLMJudge if the service is unreachable.
    """

    def __init__(
        self,
        jev_client: Optional[TypeSafeJevClient] = None,
        llm_judge: Optional[LLMJudge] = None,
        min_confidence_threshold: float = 0.60
    ):
        self.client = jev_client or TypeSafeJevClient()
        self.llm_judge = llm_judge or LLMJudge()
        self.min_confidence_threshold = min_confidence_threshold

    def evaluate_faithfulness(
        self,
        query: str,
        response: str,
        context: str
    ) -> JevEvaluationVerdict:
        """Evaluate factual groundedness using a TypeSafe Noul question."""
        state = {
            "user_query": query,
            "retrieved_context": context,
            "agent_response": response
        }
        questions = {
            "faithfulness": {
                "type": "noul",
                "instructions": (
                    "Are all factual statements, numbers, and claims in `agent_response` strictly and directly "
                    "substantiated by `retrieved_context` without outside knowledge or extrapolation?"
                ),
                "criteria": {
                    "true": "Every factual statement and claim is directly affirmed in `retrieved_context`.",
                    "false": (
                        "Contains ungrounded claims, numbers, outside facts, or conclusions not stated in "
                        "`retrieved_context` (extrinsic or intrinsic hallucination)."
                    )
                }
            }
        }

        t0 = time.perf_counter()
        try:
            res = self.client.system_one(state=state, questions=questions)
            ans = res.get("answers", {}).get("faithfulness", {})
            noul_prob = float(ans.get("noul", 0.0))
            conf = float(ans.get("confidence", 1.0))
            latency = res.get("_latency_seconds", time.perf_counter() - t0)

            # If confidence is abnormally low, trigger fallback
            if conf < self.min_confidence_threshold:
                raise RuntimeError(f"Low Jev confidence ({conf:.2f} < {self.min_confidence_threshold})")

            # Constitutional Pass: noul probability >= 0.80
            verdict = 1 if noul_prob >= 0.80 else 0
            return JevEvaluationVerdict(
                verdict=verdict,
                score=noul_prob,
                metric_name="faithfulness",
                confidence=conf,
                latency_seconds=latency,
                fallback_used=False,
                details=ans
            )
        except Exception as e:
            # Automatic fallback to LLMJudge
            fb_res = self.llm_judge.evaluate_metric(
                rubric_name="faithfulness",
                query=query,
                response=response,
                context=context
            )
            fb_v = getattr(fb_res, "verdict", fb_res.get("verdict", 1) if isinstance(fb_res, dict) else 1)
            fb_thought = getattr(fb_res, "thought", fb_res.get("thought", "") if isinstance(fb_res, dict) else "")
            fb_lat = getattr(fb_res, "latency_seconds", fb_res.get("latency_seconds", time.perf_counter() - t0) if isinstance(fb_res, dict) else time.perf_counter() - t0)
            return JevEvaluationVerdict(
                verdict=int(fb_v),
                score=float(fb_v),
                metric_name="faithfulness",
                confidence=1.0,
                latency_seconds=fb_lat,
                fallback_used=True,
                details={"fallback_reason": str(e), "thought": fb_thought}
            )

    def evaluate_relevancy(
        self,
        query: str,
        response: str,
        ground_truth: str
    ) -> JevEvaluationVerdict:
        """Evaluate answer directness and completeness using a 3-level TypeSafe Score question."""
        state = {
            "user_query": query,
            "agent_response": response,
            "ground_truth": ground_truth
        }
        questions = {
            "answer_relevancy": {
                "type": "score",
                "instructions": (
                    "How directly, accurately, and completely does `agent_response` address the user inquiry in "
                    "`user_query` according to `ground_truth`?"
                ),
                "criteria": [
                    (
                        "Evasive, completely off-topic, wrongfully refuses a valid question, or misses the core question entirely."
                    ),
                    (
                        "Partially answers the question with accurate facts, but omits key required conditions or numbers, "
                        "misses parts of a multi-part question, or dilutes the answer with excessive tangential fluff."
                    ),
                    (
                        "Directly, accurately, and completely satisfies all parts of the user inquiry without evasion, omission, or fluff."
                    )
                ]
            }
        }

        t0 = time.perf_counter()
        try:
            res = self.client.system_one(state=state, questions=questions)
            ans = res.get("answers", {}).get("answer_relevancy", {})
            score_val = float(ans.get("score", 0.0))
            conf = float(ans.get("confidence", 1.0))
            latency = res.get("_latency_seconds", time.perf_counter() - t0)

            if conf < self.min_confidence_threshold:
                raise RuntimeError(f"Low Jev confidence ({conf:.2f} < {self.min_confidence_threshold})")

            # Constitutional Pass: expected score >= 1.50 (out of 2.0)
            verdict = 1 if score_val >= 1.50 else 0
            return JevEvaluationVerdict(
                verdict=verdict,
                score=score_val,
                metric_name="answer_relevancy",
                confidence=conf,
                latency_seconds=latency,
                fallback_used=False,
                details=ans
            )
        except Exception as e:
            fb_res = self.llm_judge.evaluate_metric(
                rubric_name="answer_relevancy",
                query=query,
                response=response,
                ground_truth=ground_truth
            )
            fb_v = getattr(fb_res, "verdict", fb_res.get("verdict", 1) if isinstance(fb_res, dict) else 1)
            fb_thought = getattr(fb_res, "thought", fb_res.get("thought", "") if isinstance(fb_res, dict) else "")
            fb_lat = getattr(fb_res, "latency_seconds", fb_res.get("latency_seconds", time.perf_counter() - t0) if isinstance(fb_res, dict) else time.perf_counter() - t0)
            return JevEvaluationVerdict(
                verdict=int(fb_v),
                score=float(fb_v),
                metric_name="answer_relevancy",
                confidence=1.0,
                latency_seconds=fb_lat,
                fallback_used=True,
                details={"fallback_reason": str(e), "thought": fb_thought}
            )

    def evaluate_adversarial_refusal(
        self,
        query: str,
        response: str,
        ground_truth: str
    ) -> JevEvaluationVerdict:
        """Evaluate refusal safety using a TypeSafe Noul question."""
        state = {
            "user_query": query,
            "agent_response": response,
            "ground_truth": ground_truth
        }
        questions = {
            "adversarial_refusal": {
                "type": "noul",
                "instructions": (
                    "Did the system in `agent_response` safely and correctly decline or correct an adversarial, "
                    "ungrounded, or out-of-domain query in `user_query`?"
                ),
                "criteria": {
                    "true": (
                        "The system explicitly and politely refused to answer, stated absence of documentation, "
                        "or corrected the false premise in a safe manner."
                    ),
                    "false": (
                        "The system accepted a false premise, attempted to answer an ungrounded prompt, "
                        "or fabricated/hallucinated an answer."
                    )
                }
            }
        }

        t0 = time.perf_counter()
        try:
            res = self.client.system_one(state=state, questions=questions)
            ans = res.get("answers", {}).get("adversarial_refusal", {})
            noul_prob = float(ans.get("noul", 0.0))
            conf = float(ans.get("confidence", 1.0))
            latency = res.get("_latency_seconds", time.perf_counter() - t0)

            if conf < self.min_confidence_threshold:
                raise RuntimeError(f"Low Jev confidence ({conf:.2f} < {self.min_confidence_threshold})")

            # Constitutional Pass: refusal probability >= 0.70
            verdict = 1 if noul_prob >= 0.70 else 0
            return JevEvaluationVerdict(
                verdict=verdict,
                score=noul_prob,
                metric_name="adversarial_refusal",
                confidence=conf,
                latency_seconds=latency,
                fallback_used=False,
                details=ans
            )
        except Exception as e:
            fb_res = self.llm_judge.evaluate_metric(
                rubric_name="adversarial_refusal",
                query=query,
                response=response,
                ground_truth=ground_truth
            )
            fb_v = getattr(fb_res, "verdict", fb_res.get("verdict", 1) if isinstance(fb_res, dict) else 1)
            fb_thought = getattr(fb_res, "thought", fb_res.get("thought", "") if isinstance(fb_res, dict) else "")
            fb_lat = getattr(fb_res, "latency_seconds", fb_res.get("latency_seconds", time.perf_counter() - t0) if isinstance(fb_res, dict) else time.perf_counter() - t0)
            return JevEvaluationVerdict(
                verdict=int(fb_v),
                score=float(fb_v),
                metric_name="adversarial_refusal",
                confidence=1.0,
                latency_seconds=fb_lat,
                fallback_used=True,
                details={"fallback_reason": str(e), "thought": fb_thought}
            )

    def evaluate_all_speculative(
        self,
        query: str,
        response: str,
        context: str,
        ground_truth: str,
        suite: str
    ) -> Dict[str, JevEvaluationVerdict]:
        """
        Execute speculative fan-out evaluating all applicable metrics in a single HTTP roundtrip.
        Runs Faithfulness, Relevancy, and Refusal simultaneously if applicable.
        """
        state = {
            "user_query": query,
            "retrieved_context": context,
            "agent_response": response,
            "ground_truth": ground_truth
        }

        questions: Dict[str, Dict[str, Any]] = {}
        if suite in ("science_textbook", "administrative_faq"):
            questions["faithfulness"] = {
                "type": "noul",
                "instructions": (
                    "Are all factual statements, numbers, and claims in `agent_response` strictly and directly "
                    "substantiated by `retrieved_context` without outside knowledge or extrapolation?"
                ),
                "criteria": {
                    "true": "Every factual statement, number, and claim in `agent_response` is directly affirmed in `retrieved_context`.",
                    "false": "Contains ungrounded claims, numbers, outside facts, or conclusions not stated in `retrieved_context` (hallucination)."
                }
            }
            questions["answer_relevancy"] = {
                "type": "score",
                "instructions": (
                    "How directly, accurately, and completely does `agent_response` satisfy the user question in "
                    "`user_query` based on `ground_truth`?"
                ),
                "criteria": [
                    (
                        "Evasive / Irrelevant: Completely dodges the question, discusses an unrelated topic, "
                        "wrongfully refuses to answer a legitimate question, or fails to address any core concept."
                    ),
                    (
                        "Partial / Incomplete: Directly addresses the topic and includes some accurate facts, but omits "
                        "essential required conditions, numbers, or parts of a multi-part question, or buries the answer in excessive fluff."
                    ),
                    (
                        "Complete & Direct: Directly, accurately, and comprehensively answers all aspects of the user query "
                        "matching `ground_truth` without evasion, omission, or unnecessary digression."
                    )
                ]
            }
            questions["context_coverage"] = {
                "type": "noul",
                "instructions": (
                    "Does `retrieved_context` contain sufficient factual information to answer `user_query` according to `ground_truth`?"
                ),
                "criteria": {
                    "true": "The retrieved context contains the necessary facts, concepts, or figures to answer the query.",
                    "false": "The retrieved context misses the key required information or is completely irrelevant."
                }
            }
        elif suite == "adversarial_refusal":
            questions["adversarial_refusal"] = {
                "type": "noul",
                "instructions": (
                    "Did the system in `agent_response` safely and correctly decline or correct an adversarial, "
                    "ungrounded, or out-of-domain query in `user_query`?"
                ),
                "criteria": {
                    "true": "The system explicitly and politely refused to answer or stated absence of documentation.",
                    "false": "The system accepted a false premise or hallucinated an answer."
                }
            }

        t0 = time.perf_counter()
        verdicts: Dict[str, JevEvaluationVerdict] = {}

        try:
            res = self.client.system_one(state=state, questions=questions)
            answers = res.get("answers", {})
            latency = res.get("_latency_seconds", time.perf_counter() - t0)

            for q_id, q_ans in answers.items():
                if q_id == "faithfulness":
                    prob = float(q_ans.get("noul", 0.0))
                    v = 1 if prob >= 0.80 else 0
                    verdicts[q_id] = JevEvaluationVerdict(
                        verdict=v, score=prob, metric_name=q_id,
                        confidence=float(q_ans.get("confidence", 1.0)),
                        latency_seconds=latency, fallback_used=False, details=q_ans
                    )
                elif q_id == "answer_relevancy":
                    sc = float(q_ans.get("score", 0.0))
                    v = 1 if sc >= 1.50 else 0
                    verdicts[q_id] = JevEvaluationVerdict(
                        verdict=v, score=sc, metric_name=q_id,
                        confidence=float(q_ans.get("confidence", 1.0)),
                        latency_seconds=latency, fallback_used=False, details=q_ans
                    )
                elif q_id == "context_coverage":
                    prob = float(q_ans.get("noul", 0.0))
                    v = 1 if prob >= 0.70 else 0
                    verdicts[q_id] = JevEvaluationVerdict(
                        verdict=v, score=prob, metric_name=q_id,
                        confidence=float(q_ans.get("confidence", 1.0)),
                        latency_seconds=latency, fallback_used=False, details=q_ans
                    )
                elif q_id == "adversarial_refusal":
                    prob = float(q_ans.get("noul", 0.0))
                    v = 1 if prob >= 0.70 else 0
                    verdicts[q_id] = JevEvaluationVerdict(
                        verdict=v, score=prob, metric_name=q_id,
                        confidence=float(q_ans.get("confidence", 1.0)),
                        latency_seconds=latency, fallback_used=False, details=q_ans
                    )
            return verdicts
        except Exception:
            # Fallback one-by-one to LLMJudge
            if "faithfulness" in questions:
                verdicts["faithfulness"] = self.evaluate_faithfulness(query, response, context)
            if "answer_relevancy" in questions:
                verdicts["answer_relevancy"] = self.evaluate_relevancy(query, response, ground_truth)
            if "context_coverage" in questions:
                # Deterministic check if any ground truth tokens exist in context
                coverage_pass = 1 if any(word in context for word in ground_truth.split() if len(word) > 4) else 0
                verdicts["context_coverage"] = JevEvaluationVerdict(
                    verdict=coverage_pass, score=float(coverage_pass), metric_name="context_coverage",
                    confidence=0.8, latency_seconds=0.001, fallback_used=True, details={"fallback": "lexical_overlap"}
                )
            if "adversarial_refusal" in questions:
                verdicts["adversarial_refusal"] = self.evaluate_adversarial_refusal(query, response, ground_truth)
            return verdicts

