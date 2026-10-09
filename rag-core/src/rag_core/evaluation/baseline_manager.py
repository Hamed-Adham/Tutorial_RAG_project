"""
Stage 10: Baseline Snapshot Manager

Standalone module for saving, loading, and comparing RAGAS evaluation
baselines. Extracted from run_eval.py for reuse by harness, CI scripts,
and notebook workflows.

Baselines are stored as JSON snapshots in:
  rag-core/src/rag_core/evaluation/baselines/

The ``baseline_latest.json`` file always reflects the most recently
promoted baseline. Historical snapshots use timestamp-suffixed filenames.
"""

from __future__ import annotations

# Re-export the BaselineManager from run_eval for backwards compatibility
# and direct use without needing to import the full CLI module.

from rag_core.evaluation.run_eval import BaselineManager

__all__ = ["BaselineManager"]
