#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ragas Evaluation Adapter for Unified RAG.
Connects Ragas metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
to the local Nine Router OpenAI-compatible gateway.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from rag_core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class RagasEvaluationReport:
    metrics: Dict[str, float]
    sample_scores: List[Dict[str, Any]]
    latency_seconds: float
    total_samples: int
    error: Optional[str] = None

class LocalRagasEvaluator:
    """Wrapper configuring Ragas with local Nine Router LLM and BGE-M3 embeddings."""

    def __init__(
        self,
        base_url: str = settings.nine_router_base_url,
        api_key: str = settings.nine_router_api_key,
        model: str = settings.llm_model,
        embedding_model: str = settings.embedding_model_name
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model

    def evaluate_samples(self, samples: List[Dict[str, Any]]) -> RagasEvaluationReport:
        """
        Evaluate a list of samples.
        Each sample dict must have:
        - "question": str (User query)
        - "contexts": List[str] (Retrieved text chunks)
        - "answer": str (Synthesized answer)
        - "ground_truth": Optional[str] (Reference ground truth)
        """
        t0 = time.time()
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            # Initialize LangChain wrappers for Nine Router
            evaluator_llm = ChatOpenAI(
                model=self.model,
                openai_api_base=self.base_url,
                openai_api_key=self.api_key,
                temperature=0.0,
                max_retries=2,
                timeout=60
            )

            evaluator_embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_base=self.base_url,
                openai_api_key=self.api_key
            )

            # Format data for Ragas schema
            formatted_data = {
                "question": [s["question"] for s in samples],
                "contexts": [s.get("contexts", []) for s in samples],
                "answer": [s.get("answer", "") for s in samples],
                "ground_truth": [s.get("ground_truth", "") for s in samples]
            }

            dataset = Dataset.from_dict(formatted_data)
            
            # Select metrics (if no ground truth, evaluate reference-free subset)
            has_gt = any(bool(s.get("ground_truth")) for s in samples)
            metrics_to_run = [faithfulness, answer_relevancy]
            if has_gt:
                metrics_to_run.extend([context_precision, context_recall])

            result = evaluate(
                dataset=dataset,
                metrics=metrics_to_run,
                llm=evaluator_llm,
                embeddings=evaluator_embeddings
            )

            elapsed = round(time.time() - t0, 3)
            metric_scores = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}

            return RagasEvaluationReport(
                metrics=metric_scores,
                sample_scores=result.to_pandas().to_dict(orient="records"),
                latency_seconds=elapsed,
                total_samples=len(samples)
            )

        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            logger.error("Ragas evaluation failed: %s", e)
            return RagasEvaluationReport(
                metrics={},
                sample_scores=[],
                latency_seconds=elapsed,
                total_samples=len(samples),
                error=str(e)
            )
