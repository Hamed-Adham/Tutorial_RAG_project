#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation REST Endpoints (Feature 007 - US6).
Provides endpoints for triggering benchmark suites, monitoring execution progress,
and fetching diagnostic confusion matrices and reports.
"""

from typing import List
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import PlainTextResponse

from app.schemas.evaluation import (
    EvaluationRunRequest,
    EvaluationRunResponse,
    EvaluationRunStatus,
    SuiteListResponse,
)
from app.services.evaluation_service import EvaluationService
from rag_core.evaluation.report_generator import ReportGenerator

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.get("/suites", response_model=SuiteListResponse, summary="List available benchmark suites")
def list_suites():
    """Retrieve metadata and item distributions for all 5 benchmark suites."""
    return EvaluationService.list_suites()


@router.post("/run", response_model=EvaluationRunResponse, status_code=status.HTTP_202_ACCEPTED, summary="Trigger a benchmark evaluation run")
def start_evaluation_run(request: EvaluationRunRequest):
    """
    Launch a background evaluation run for a target suite with chosen scope and options.
    """
    try:
        return EvaluationService.start_run(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate evaluation run: {str(e)}"
        )


@router.get("/status/{run_id}", response_model=EvaluationRunStatus, summary="Poll evaluation progress and results")
def get_evaluation_status(run_id: str):
    """
    Fetch the live status, progress percentage, current item, and report of an evaluation run.
    """
    status_entry = EvaluationService.get_status(run_id)
    if not status_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run '{run_id}' not found."
        )
    return status_entry


@router.get("/runs", response_model=List[EvaluationRunStatus], summary="List all past and active runs")
def list_evaluation_runs():
    """List all registered benchmark runs."""
    return EvaluationService.list_runs()


@router.get("/report/{run_id}/markdown", response_class=PlainTextResponse, summary="Download executive Markdown report")
def download_report_markdown(run_id: str):
    """
    Export the evaluation report for run_id as publication-ready GitHub-flavored Markdown.
    """
    status_entry = EvaluationService.get_status(run_id)
    if not status_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run '{run_id}' not found."
        )
    if not status_entry.report:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Evaluation run '{run_id}' has not produced a report yet (status: {status_entry.status})."
        )

    generator = ReportGenerator()
    return generator.generate_markdown(status_entry.report)
