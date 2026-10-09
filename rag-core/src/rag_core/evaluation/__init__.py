"""
Evaluation & Benchmarking subsystem for Unified RAG.
Includes golden test suites, 5-part judge rubrics, and meta-evaluation reliability harness.
"""

from rag_core.evaluation.rubrics import (
    JudgeRubric,
    RUBRICS,
    get_rubric,
    FAITHFULNESS_RUBRIC,
    ANSWER_RELEVANCY_RUBRIC,
    DOMAIN_ROUTING_RUBRIC,
    ADVERSARIAL_REFUSAL_RUBRIC,
)
from rag_core.evaluation.meta_eval import (
    ConfusionMatrix,
    compute_confusion_matrix,
    ReliabilityMetrics,
    calculate_pass_power_k,
    calculate_pass_at_k,
    evaluate_reliability_harness,
)
from rag_core.evaluation.datasets.load_dataset import (
    load_golden_dataset,
    validate_dataset_integrity,
)
from rag_core.evaluation.judge_evaluator import LLMJudge
from rag_core.evaluation.jev_judge import JevJudge


__all__ = [
    "JudgeRubric",
    "RUBRICS",
    "get_rubric",
    "FAITHFULNESS_RUBRIC",
    "ANSWER_RELEVANCY_RUBRIC",
    "DOMAIN_ROUTING_RUBRIC",
    "ADVERSARIAL_REFUSAL_RUBRIC",
    "ConfusionMatrix",
    "compute_confusion_matrix",
    "ReliabilityMetrics",
    "calculate_pass_power_k",
    "calculate_pass_at_k",
    "evaluate_reliability_harness",
    "load_golden_dataset",
    "validate_dataset_integrity",
    "LLMJudge",
    "JevJudge",
]
