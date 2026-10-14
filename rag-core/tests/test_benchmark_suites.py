#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit and boundary tests for Feature 007 calibrated benchmark suites.
Verifies total item count (360), suite distributions, difficulty ratios (20% Easy, 50% Med, 30% Hard),
and sealed 25% dev (90 items) / 75% test (270 items) splits.
"""

import pytest
from collections import Counter
from pathlib import Path

from rag_core.evaluation.benchmark_schema import BenchmarkItem, SuiteType, SplitType, DifficultyLevel
from rag_core.evaluation.datasets.load_dataset import load_benchmark_items


def test_master_golden_dataset_total_count_and_suites():
    """Verify master golden benchmark has exactly 360 items across 5 suites."""
    items = load_benchmark_items()
    assert len(items) == 360, f"Expected 360 items, got {len(items)}"

    suites_count = Counter(it.suite for it in items)
    assert suites_count[SuiteType.SCIENCE_CURRICULUM] == 120
    assert suites_count[SuiteType.ADMINISTRATIVE_FAQ] == 80
    assert suites_count[SuiteType.CONVERSATIONAL_MEMORY] == 60
    assert suites_count[SuiteType.SUBQUERY_DECOMPOSITION] == 50
    assert suites_count[SuiteType.ADVERSARIAL_REFUSAL] == 50


def test_sealed_dev_and_test_splits():
    """Verify exact 25% dev (90 items) and 75% test (270 items) partition."""
    dev_items = load_benchmark_items(split="dev")
    test_items = load_benchmark_items(split="test")

    assert len(dev_items) == 90, f"Expected 90 dev items, got {len(dev_items)}"
    assert len(test_items) == 270, f"Expected 270 test items, got {len(test_items)}"

    dev_ids = {it.item_id for it in dev_items}
    test_ids = {it.item_id for it in test_items}

    # Zero leakage between dev and test
    intersection = dev_ids.intersection(test_ids)
    assert len(intersection) == 0, f"Found leaking items between dev and test: {intersection}"


def test_controlled_difficulty_distribution():
    """Verify non-saturating difficulty distribution (~20% Easy, ~50% Med, ~30% Hard/Adversarial)."""
    items = load_benchmark_items()
    diff_counts = Counter(it.difficulty for it in items)

    easy_pct = (diff_counts[DifficultyLevel.EASY]) / len(items)
    med_pct = (diff_counts[DifficultyLevel.MEDIUM]) / len(items)
    hard_pct = (diff_counts[DifficultyLevel.HARD] + diff_counts[DifficultyLevel.ADVERSARIAL]) / len(items)

    assert 0.15 <= easy_pct <= 0.25, f"Easy pct out of bounds: {easy_pct:.2f}"
    assert 0.45 <= med_pct <= 0.55, f"Medium pct out of bounds: {med_pct:.2f}"
    assert 0.25 <= hard_pct <= 0.35, f"Hard/Adversarial pct out of bounds: {hard_pct:.2f}"


def test_conversational_memory_suite_structure():
    """Verify conversational memory items contain historical turns with is_needed flag."""
    memory_items = load_benchmark_items(suite="conversational_memory")
    assert len(memory_items) == 60

    for it in memory_items:
        assert len(it.history_turns) >= 2, f"Item {it.item_id} has fewer than 2 turns"
        needed_turns = [t for t in it.history_turns if t.is_needed]
        assert len(needed_turns) >= 1, f"Item {it.item_id} has no needed turns"


def test_adversarial_refusal_suite_ground_truth():
    """Verify adversarial refusal items specify standardized Persian refusal formula."""
    refusal_items = load_benchmark_items(suite="adversarial_refusal")
    assert len(refusal_items) == 50

    standard_refusal = "اطلاعات مربوط به این پرسش در مستندات موجود نمی‌باشد"
    for it in refusal_items:
        assert it.expected_behavior == "refusal"
        assert standard_refusal in it.ground_truth
