from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatStreamRequest
from app.services.groq_client import GroqClient

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream")
async def stream_chat(
    request: ChatStreamRequest = Body(...)
) -> StreamingResponse:
    client = GroqClient()

    async def generator():
        async for chunk in client.stream_chat_completion(
            messages=[m.model_dump() for m in request.messages],
            model=request.model,
        ):
            yield chunk.encode("utf-8")

    return StreamingResponse(
        generator(),
        media_type="text/plain; charset=utf-8",
    )
