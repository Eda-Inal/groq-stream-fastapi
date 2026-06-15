from collections.abc import AsyncIterator

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.groq_client import LLMClient
from app.services.tool_client.remote_client import RemoteToolClient

logger = structlog.get_logger()


class ReActAgentService:
    """
    Orchestrates a text-based ReAct (Thought/Action/Observation/Final Answer)
    loop. Independent from ChatService: no native function-calling, no shared
    state with the /chat/stream pipeline.
    """

    def __init__(self) -> None:
        self.client = LLMClient()
        self.mcp = RemoteToolClient()

    async def stream_agent(
        self,
        *,
        session: AsyncSession,
        messages: list[dict],
        model: str,
        user_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        conversation_id: str | None = None,
        tags: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        """
        Run the ReAct loop and yield streaming events.

        Stage 0 placeholder — the Thought/Action/Observation loop is built
        out in later stages.
        """
        logger.info(
            "agent_stream_not_implemented",
            model=model,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        yield {"type": "thought", "text": "Agent not implemented yet."}
        yield {"type": "done", "finish_reason": "stop"}
