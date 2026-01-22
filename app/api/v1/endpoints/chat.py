"""
Chat API endpoints.

This module exposes HTTP endpoints for interacting with
the Groq streaming chat service.
"""

from typing import AsyncIterator
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import structlog

from app.schemas.chat import ChatStreamRequest
from app.services.groq_client import GroqClient, GroqStreamError

router = APIRouter(prefix="/chat", tags=["chat"])
logger = structlog.get_logger()


@router.post("/stream")
async def stream_chat(request: ChatStreamRequest) -> StreamingResponse:
    """
    Stream a chat completion response from Groq to the client.

    The response is sent incrementally as JSON events
    using Server-Sent Events (SSE).
    """
    log = logger.bind(
        message_count=len(request.messages),
        model_override=request.model,
    )
    log.info("chat_stream_request_received")

    client = GroqClient()

    try:
        generator = client.stream_chat_completion(
            messages=[m.model_dump() for m in request.messages],
            model=request.model,
        )
    except GroqStreamError as e:
        log.error("chat_stream_initialization_failed", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))

    async def stream_generator() -> AsyncIterator[bytes]:
        try:
            async for event in generator:
                payload = json.dumps(event, ensure_ascii=False)
                yield f"data: {payload}\n\n".encode("utf-8")

            log.info("chat_stream_completed_successfully")

        except Exception as e:
            log.error(
                "chat_stream_broken_midway",
                error=str(e),
                exc_info=True,
            )
            error_event = {
                "type": "error",
                "message": "Stream interrupted",
            }
            yield f"data: {json.dumps(error_event)}\n\n".encode("utf-8")

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
    )
