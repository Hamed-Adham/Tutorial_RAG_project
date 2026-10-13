"""
Chat Endpoints: Batch execution and Server-Sent Events (SSE) token streaming.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import rag_service, RAGService

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Execute synchronous batch RAG inquiry")
async def chat_endpoint(
    request: ChatRequest,
    service: RAGService = Depends(lambda: rag_service)
) -> ChatResponse:
    """
    Submits a natural language science inquiry.
    Processes query within concurrency slot, loads multi-turn session history,
    resolves antecedents, and returns grounded answer with page citations.
    """
    return await service.execute_chat(request)


@router.post("/chat/stream", summary="Stream real-time tokens via Server-Sent Events (SSE)")
async def chat_stream_endpoint(
    request: ChatRequest,
    service: RAGService = Depends(lambda: rag_service)
):
    """
    Submits a science inquiry and returns a live Server-Sent Events stream.
    Emits token events progressively (TTFT < 1.0s) and closes with verified citations.
    """
    return StreamingResponse(
        service.execute_chat_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
