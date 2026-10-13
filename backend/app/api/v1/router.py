"""
API v1 Router aggregating all endpoints.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import chat, sessions, health, metrics

api_router = APIRouter()

api_router.include_router(chat.router, tags=["Chat"])
api_router.include_router(sessions.router, tags=["Sessions"])
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(metrics.router, tags=["Metrics"])
