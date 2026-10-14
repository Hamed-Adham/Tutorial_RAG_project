#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for expanded Layer 1 Deterministic Code Gates (Feature 007).
Validates Persian refusal regex, retrieval Hit@k and MRR, citation format, and sub-5ms latencies.
"""

import pytest
from rag_core.evaluation.code_gates import (
    evaluate_refusal_gate,
    evaluate_retrieval_hit_gate,
    evaluate_citation_gate,
    evaluate_domain_routing_gate,
    evaluate_code_gate,
    is_standardized_refusal,
    has_valid_citation,
)


def test_persian_refusal_patterns():
    """Verify standard and regex refusal patterns in Persian."""
    valid_refusals = [
        "اطلاعات مربوط به این پرسش در مستندات موجود نمی‌باشد",
        "با عرض پوزش، اطلاعاتی در این زمینه یافت نشد.",
        "در مستندات ارائه شده پاسخی برای این پرسش وجود ندارد.",
        "متأسفانه پاسخ این پرسش در پایگاه داده موجود نمی‌باشد.",
        "صندوق نوآوری هیچ‌گونه حمایتی برای وام شخصی ارائه نمی‌دهد.",
        "چنین مطلبی در کتاب علوم تجربی هفتم ذکر نشده است.",
    ]
    for text in valid_refusals:
        assert is_standardized_refusal(text) is True, f"Failed on: {text}"
        res = evaluate_refusal_gate(text)
        assert res.passed == 1
        assert res.latency_ms < 5.0

    non_refusal = "بطن چپ خون روشن را به تمام بدن پمپاژ می‌کند."
    assert is_standardized_refusal(non_refusal) is False
    res = evaluate_refusal_gate(non_refusal)
    assert res.passed == 0


def test_retrieval_hit_and_mrr_calculation():
    """Verify deterministic Hit@k and MRR calculation."""
    gold = ["chunk_sci_001", "chunk_sci_002"]

    # Rank 1 hit: Hit@5 = 1, MRR = 1.0
    retrieved_r1 = ["chunk_sci_001", "other_1", "other_2"]
    res = evaluate_retrieval_hit_gate(gold, retrieved_r1, k=5)
    assert res.passed == 1
    assert res.hit_at_k == 1
    assert res.mrr == 1.0

    # Rank 3 hit: Hit@5 = 1, MRR = 1/3 ~ 0.33
    retrieved_r3 = ["other_1", "other_2", "chunk_sci_002", "other_3"]
    res = evaluate_retrieval_hit_gate(gold, retrieved_r3, k=5)
    assert res.passed == 1
    assert res.hit_at_k == 1
    assert round(res.mrr, 2) == 0.33

    # Miss: Hit@5 = 0, MRR = 0.0
    retrieved_miss = ["other_1", "other_2", "other_3"]
    res = evaluate_retrieval_hit_gate(gold, retrieved_miss, k=5)
    assert res.passed == 0
    assert res.hit_at_k == 0
    assert res.mrr == 0.0


def test_citation_formatting_gate():
    """Verify inline citation bracket detection."""
    valid_citation_answers = [
        "بطن چپ دارای دیواره ضخیم‌تری است [C706_olom_haftom:124].",
        "تسهیلات شرکت‌های دانش‌بنیان طبق [faq_qa_pairs.csv#row=1] تعیین می‌گردد.",
        "گوارش مکانیکی در دهان آغاز می‌شود (منبع: فصل 13).",
    ]
    for ans in valid_citation_answers:
        assert has_valid_citation(ans) is True
        res = evaluate_citation_gate(ans)
        assert res.passed == 1

    uncited_answer = "بطن چپ دارای دیواره ضخیم‌تری است بدون منبع."
    res = evaluate_citation_gate(uncited_answer, requires_citations=True)
    assert res.passed == 0


def test_domain_routing_keyword_gate():
    """Verify deterministic domain routing by Persian keywords."""
    sci_q = "ساختار یاخته و میتوکندری چیست؟"
    res = evaluate_domain_routing_gate(sci_q, expected_route="science_curriculum")
    assert res.passed == 1

    admin_q = "شرایط دریافت ضمانت‌نامه حسن انجام کار در سامانه غزال چیست؟"
    res = evaluate_domain_routing_gate(admin_q, expected_route="administrative_faq")
    assert res.passed == 1
