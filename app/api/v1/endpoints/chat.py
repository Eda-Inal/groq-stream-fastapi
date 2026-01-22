"""
Chat API endpoints.

This module exposes HTTP endpoints for interacting with
the Groq streaming chat service.
"""

from typing import AsyncIterator
import json
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import structlog

from app.schemas.chat import ChatStreamRequest
from app.services.groq_client import GroqClient, GroqStreamError

router = APIRouter(prefix="/chat", tags=["chat"])
logger = structlog.get_logger()


@router.post("/stream")
async def stream_chat(payload: ChatStreamRequest, request: Request) -> StreamingResponse:
    """
    Stream a chat completion response from Groq to the client.

    The response is sent incrementally as OpenAI-compatible
    chat.completion.chunk events using Server-Sent Events (SSE).
    """
    log = logger.bind(
        message_count=len(payload.messages),
        model_override=payload.model,
    )
    log.info("chat_stream_request_received")

    client = GroqClient()

    try:
        generator = client.stream_chat_completion(
            messages=[m.model_dump() for m in payload.messages],
            model=payload.model,
        )
    except GroqStreamError as e:
        log.error("chat_stream_initialization_failed", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))


    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_name = payload.model or "llama-3.3-70b-versatile"

    async def stream_generator() -> AsyncIterator[bytes]:
        try:
            role_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n".encode("utf-8")


            async for event in generator:
                if await request.is_disconnected():
                    log.info("chat_stream_cancelled_by_client")
                    break

                if event.get("type") == "chunk":
                    content_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": event["text"]},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n".encode("utf-8")

                elif event.get("type") == "done":
                    final_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": event.get("finish_reason", "stop"),
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8")
                    break

            yield b"data: [DONE]\n\n"
            log.info("chat_stream_completed_successfully")

        except Exception as e:
            log.error(
                "chat_stream_broken_midway",
                error=str(e),
                exc_info=True,
            )
            error_chunk = {
                "id": completion_id,
                "object": "error",
                "message": "Stream interrupted",
            }
            yield f"data: {json.dumps(error_chunk)}\n\n".encode("utf-8")

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
    )