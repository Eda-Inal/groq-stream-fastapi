from typing import AsyncIterator
import json

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import structlog

from app.schemas.research import ResearchRequest
from app.services.research_service import ResearchService
from app.core.config import AVAILABLE_MODELS, settings

router = APIRouter(prefix="/research", tags=["research"])
logger = structlog.get_logger()

research_service = ResearchService()


@router.post("/stream")
async def stream_research(
    payload: ResearchRequest,
    request: Request,
) -> StreamingResponse:
    model = payload.model or settings.groq_default_model

    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    if not settings.tavily_api_key:
        raise HTTPException(status_code=503, detail="Web search is not available: missing Tavily API key.")

    log = logger.bind(
        topic_length=len(payload.topic),
        model=model,
        user_id=payload.user_id,
        language=payload.language,
    )
    log.info("research_request_received")

    generator = research_service.stream_research(
        topic=payload.topic,
        model=model,
        user_id=payload.user_id,
        language=payload.language,
    )

    async def stream_generator() -> AsyncIterator[bytes]:
        try:
            async for event in generator:
                if await request.is_disconnected():
                    log.info("research_stream_cancelled_by_client")
                    break
                yield f"data: {json.dumps(event)}\n\n".encode("utf-8")

            yield b"data: [DONE]\n\n"
            log.info("research_stream_completed_successfully")

        except Exception as e:
            log.error("research_stream_broken_midway", error=str(e), exc_info=True)
            error_event = {"type": "error", "message": "Stream interrupted", "phase": "endpoint"}
            yield f"data: {json.dumps(error_event)}\n\n".encode("utf-8")

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
