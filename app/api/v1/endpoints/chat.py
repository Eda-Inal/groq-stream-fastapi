"""
Chat API endpoints.

This module exposes HTTP endpoints for interacting with
the Groq streaming chat service.
"""

from typing import AsyncIterator
import json
import time
import uuid

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.chat_bulk import BulkChatRequest
from app.schemas.chat import ChatStreamRequest
from app.db.session import get_db
from app.services.chat_service import ChatService
from app.core.config import AVAILABLE_MODELS, settings

router = APIRouter(prefix="/chat", tags=["chat"])
logger = structlog.get_logger()

chat_service = ChatService()


@router.get("/models")
async def list_models():
    """
    OpenAI-compatible models endpoint.

    Returns the list of available LLM models that the client can select.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
            }
            for model_id in AVAILABLE_MODELS.keys()
        ],
    }


@router.post("/stream")
async def stream_chat(
    payload: ChatStreamRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
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

    # --- Model resolution & validation (OpenAI-style) ---
    model = payload.model or settings.groq_default_model

    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model}",
        )

    generator = chat_service.stream_chat(
        session=db,
        messages=[m.model_dump() for m in payload.messages],
        model=model,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        top_p=payload.top_p,
        frequency_penalty=payload.frequency_penalty,
        presence_penalty=payload.presence_penalty,
        stop=payload.stop,
        seed=payload.seed,
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_name = model

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

@router.post("/bulk")
async def bulk_chat(
    payload: BulkChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Bulk non-streaming chat endpoint.
    """

    results = await chat_service.bulk_complete(
        session=db,
        items=payload.items,
    )

    return {
        "total": len(payload.items),
        "results": results,
    }