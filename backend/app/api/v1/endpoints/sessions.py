"""
Session Endpoints: Inspect and delete multi-turn conversation memory.
"""

from fastapi import APIRouter, HTTPException, Depends
from app.schemas.session import SessionStateModel, DeleteSessionResponse
from app.services.session_store import get_session_store, ISessionStore

router = APIRouter()


@router.get("/sessions/{session_id}", response_model=SessionStateModel, summary="Get conversation history for session")
async def get_session_endpoint(
    session_id: str,
    store: ISessionStore = Depends(get_session_store)
) -> SessionStateModel:
    """Fetch stored dialogue turns and metadata for a specific session ID."""
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or expired.")
    return session


@router.delete("/sessions/{session_id}", response_model=DeleteSessionResponse, summary="Purge conversation history")
async def delete_session_endpoint(
    session_id: str,
    store: ISessionStore = Depends(get_session_store)
) -> DeleteSessionResponse:
    """Purge active session history upon conversation reset or user sign-out."""
    deleted = await store.delete_session(session_id)
    return DeleteSessionResponse(session_id=session_id, deleted=deleted)
