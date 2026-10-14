#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for Evaluation REST Endpoints (Feature 007 - US6).
Tests suite catalog, run dispatch, live status polling, and markdown export.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.schemas.evaluation import EvaluationRunRequest
from rag_core.evaluation.benchmark_schema import SuiteType, SplitType, EvaluationScope


@pytest.mark.asyncio
async def test_get_evaluation_suites(async_client: AsyncClient):
    """Verify suite catalog endpoint returns all calibrated benchmark suites."""
    response = await async_client.get("/api/v1/evaluation/suites")
    assert response.status_code == 200
    data = response.json()
    assert "suites" in data
    suites = data["suites"]
    assert len(suites) >= 5

    suite_ids = [s["id"] for s in suites]
    assert "all" in suite_ids
    assert "science_curriculum" in suite_ids
    assert "administrative_faq" in suite_ids
    assert "conversational_memory" in suite_ids
    assert "subquery_decomposition" in suite_ids
    assert "adversarial_refusal" in suite_ids


@pytest.mark.asyncio
async def test_start_evaluation_run_and_poll_status(async_client: AsyncClient):
    """Verify dispatching an evaluation run and polling its status."""
    payload = {
        "suite": "adversarial_refusal",
        "split": "dev",
        "scope": "gates_only",
        "judge_model": "subq",
        "limit": 3,
    }
    # Step 1: Dispatch
    post_res = await async_client.post("/api/v1/evaluation/run", json=payload)
    assert post_res.status_code == 202
    post_data = post_res.json()
    assert "run_id" in post_data
    assert post_data["status"] in ("queued", "running")
    run_id = post_data["run_id"]

    # Step 2: Poll status immediately
    poll_res = await async_client.get(f"/api/v1/evaluation/status/{run_id}")
    assert poll_res.status_code == 200
    poll_data = poll_res.json()
    assert poll_data["run_id"] == run_id
    assert poll_data["status"] in ("queued", "running", "completed")
    assert "progress_percentage" in poll_data


@pytest.mark.asyncio
async def test_evaluation_status_not_found(async_client: AsyncClient):
    """Verify 404 response on non-existent evaluation run."""
    response = await async_client.get("/api/v1/evaluation/status/non-existent-id-999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_list_evaluation_runs(async_client: AsyncClient):
    """Verify listing all evaluation runs."""
    response = await async_client.get("/api/v1/evaluation/runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
