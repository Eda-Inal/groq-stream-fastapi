"""
Chat API endpoints.

This module exposes HTTP endpoints for interacting with
the Groq streaming chat service.
"""

from typing import AsyncIterator

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
import structlog
from app.schemas.chat import ChatStreamRequest
from app.services.groq_client import GroqClient, GroqStreamError

router = APIRouter(prefix="/chat", tags=["chat"])
logger = structlog.get_logger()

@router.post("/stream")
async def stream_chat(
    request: ChatStreamRequest # Body(...) kaldırıldı, otomatik olarak gövdeden okur
) -> StreamingResponse:
    """
    Stream a chat completion response from Groq to the client.

    The response is sent incrementally as plain text chunks
    using FastAPI's StreamingResponse.
    """
    log = logger.bind(message_count=len(request.messages), model_override=request.model)
    log.info("chat_stream_request_received")

    client = GroqClient()
    
    generator_instance = client.stream_chat_completion(
        messages=[m.model_dump() for m in request.messages],
        model=request.model,
    )

    try:
        first_chunk = await generator_instance.__anext__()
    except GroqStreamError as e:
        log.error("chat_stream_initialization_failed", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))
    except StopAsyncIteration:
        log.warning("chat_stream_empty_response")
        return StreamingResponse(iter([]), media_type="text/plain")
    except Exception as e:
        log.error("endpoint_unexpected_error_before_stream", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

    async def stream_generator() -> AsyncIterator[bytes]:
        yield first_chunk.encode("utf-8")
        try:
            async for chunk in generator_instance:
                yield chunk.encode("utf-8")
            log.info("chat_stream_completed_successfully")
        except Exception as e:
            log.error("chat_stream_broken_midway", error=str(e), exc_info=True)
            yield b"\n[ERROR] Stream interrupted"

    return StreamingResponse(
        stream_generator(),
        media_type="text/plain; charset=utf-8",
    )