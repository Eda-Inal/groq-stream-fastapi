"""
ReAct agent API endpoint.

This is a separate streaming endpoint from /chat/stream, using a
text-based ReAct (Thought/Action/Observation/Final Answer) loop instead of
native function-calling. The SSE event stream is intentionally NOT
OpenAI-compatible: each event is a JSON object with a `type` field
(`thought`, `action`, `observation`, `chunk`, `done`).
"""

from typing import AsyncIterator
import json
import uuid

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.agent import AgentStreamRequest
from app.db.session import get_db
from app.services.react_agent_service import ReActAgentService
from app.core.config import AVAILABLE_MODELS, settings

router = APIRouter(prefix="/agent", tags=["agent"])
logger = structlog.get_logger()

agent_service = ReActAgentService()


@router.post("/stream")
async def stream_agent(
    payload: AgentStreamRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Stream a ReAct agent run as custom SSE events.

    Each event is `data: {"type": ..., ...}\\n\\n`. Event types: `thought`,
    `action`, `observation`, `chunk` (final answer tokens), `done`.
    """
    conversation_id = payload.conversation_id or uuid.uuid4().hex
    model = payload.model or settings.agent_default_model

    log = logger.bind(
        message_count=len(payload.messages),
        model_override=payload.model,
        conversation_id=conversation_id,
        user_id=payload.user_id,
    )
    log.info("agent_stream_request_received")

    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model}",
        )

    generator = agent_service.stream_agent(
        session=db,
        messages=[m.model_dump() for m in payload.messages],
        model=model,
        user_id=payload.user_id,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        conversation_id=conversation_id,
        tags=payload.tags,
    )

    async def stream_generator() -> AsyncIterator[bytes]:
        try:
            async for event in generator:
                if await request.is_disconnected():
                    log.info("agent_stream_cancelled_by_client")
                    break

                yield f"data: {json.dumps(event)}\n\n".encode("utf-8")

            yield b"data: [DONE]\n\n"
            log.info("agent_stream_completed_successfully")

        except Exception as e:
            log.error(
                "agent_stream_broken_midway",
                error=str(e),
                exc_info=True,
            )
            error_event = {"type": "error", "message": "Stream interrupted"}
            yield f"data: {json.dumps(error_event)}\n\n".encode("utf-8")

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
    )
