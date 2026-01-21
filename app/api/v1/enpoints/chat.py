"""
Chat API endpoints.

This module exposes HTTP endpoints for interacting with
the Groq streaming chat service.
"""

from typing import AsyncIterator

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatStreamRequest
from app.services.groq_client import GroqClient

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream")
async def stream_chat(
    request: ChatStreamRequest = Body(...)
) -> StreamingResponse:
    """
    Stream a chat completion response from Groq to the client.

    The response is sent incrementally as plain text chunks
    using FastAPI's StreamingResponse.
    """
    client = GroqClient()

    async def generator() -> AsyncIterator[bytes]:
        """
        Async generator that forwards streamed text chunks
        from the Groq client to the HTTP response.
        """
        async for chunk in client.stream_chat_completion(
            messages=[m.model_dump() for m in request.messages],
            model=request.model,
        ):
            yield chunk.encode("utf-8")

    return StreamingResponse(
        generator(),
        media_type="text/plain; charset=utf-8",
    )
