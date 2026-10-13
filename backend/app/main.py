"""
FastAPI Main Application Factory.
Provides asynchronous gateway with CORS, lifespan startup warmup, and API v1 routing.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1.router import api_router
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup warmup and shutdown."""
    logger.info("Initializing %s...", settings.app_name)
    # Eagerly warm up pipeline models to eliminate first-query latency spike
    try:
        rag_service.pipeline.warmup()
        logger.info("AdvancedRAGPipeline models warmed up successfully.")
    except Exception as e:
        logger.warning("Pipeline warmup encountered error: %s (will load lazily)", e)

    yield

    logger.info("Shutting down %s.", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description="Asynchronous Enterprise Science RAG Gateway with Concurrency Protection & Multi-Turn Session Memory",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/", summary="Root status check")
async def root():
    return {
        "service": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health",
        "metrics": f"{settings.api_v1_prefix}/metrics"
    }
