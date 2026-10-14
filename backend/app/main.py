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


from fastapi.responses import FileResponse
from fastapi import HTTPException
from pathlib import Path

DATA_PDF_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "Olum_7_splitted"
FRONTEND_PUBLIC_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "public"


@app.get("/api/v1/pdf/{filename}", summary="Serve split textbook chapter PDF")
@app.get("/{filename}.pdf", summary="Serve split textbook chapter PDF at root")
async def serve_chapter_pdf(filename: str):
    """Serve textbook PDF with proper Content-Type application/pdf."""
    clean_name = filename.split("/")[-1]
    if not clean_name.endswith(".pdf"):
        clean_name = f"{clean_name}.pdf"

    file_path = DATA_PDF_DIR / clean_name
    if not file_path.exists():
        file_path = FRONTEND_PUBLIC_DIR / clean_name

    if not file_path.exists():
        # Fallback: check if filename references chapter number e.g. فصل 15 or ch15
        import re
        from urllib.parse import unquote
        decoded = unquote(filename)
        ch_match = re.search(r"(?:فصل\s*|chapter\s*|ch\s*)?(\d+)", decoded)
        if ch_match:
            ch_num = int(ch_match.group(1))
            from rag_core.postprocessing.citations import DOC_TO_CHAPTER
            for fname, (c_num, _) in DOC_TO_CHAPTER.items():
                if c_num == ch_num:
                    cand_path = DATA_PDF_DIR / fname
                    if not cand_path.exists():
                        cand_path = FRONTEND_PUBLIC_DIR / fname
                    if cand_path.exists():
                        file_path = cand_path
                        clean_name = fname
                        break

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF {clean_name} not found")

    return FileResponse(file_path, media_type="application/pdf")


@app.get("/", summary="Root status check")
async def root():
    return {
        "service": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health",
        "metrics": f"{settings.api_v1_prefix}/metrics"
    }
