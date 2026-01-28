from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.chat_log import create_chat_log
from app.services.groq_client import GroqClient
import asyncio
from asyncio import Semaphore
from app.schemas.chat_bulk import BulkChatItem
from app.core.config import settings


class ChatService:
    """
    Orchestrates LLM streaming and persistence.

    - Streams chunks to the caller
    - Accumulates final response
    - Persists OpenAI-compatible input + output
      even if the stream is interrupted
    """

    def __init__(self) -> None:
        self._client = GroqClient()

    async def stream_chat(
        self,
        *,
        session: AsyncSession,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[dict]:
        full_response: list[str] = []

        try:
            async for event in self._client.stream_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                seed=seed,
            ):
                if event.get("type") == "chunk":
                    full_response.append(event["text"])

                yield event

        finally:
            # Persist ONCE, even if the stream was interrupted
            if full_response:
                await create_chat_log(
                    session=session,
                    prompt=self._extract_user_prompt(messages),
                    messages=messages,
                    response="".join(full_response),
                    model_name=model or "default",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    seed=seed,
                )

    @staticmethod
    def _extract_user_prompt(messages: list[dict[str, str]]) -> str:
        """
        Extract the first user message as a short prompt summary.

        This is for quick inspection / filtering.
        The full source of truth is `messages`.
        """
        for message in messages:
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    async def bulk_complete(
        self,
        *,
        session,
        items,
        concurrency: int = 5,
    ):
        import asyncio
        from asyncio import Semaphore
        from app.core.config import settings

        semaphore = Semaphore(concurrency)
        results = []

        async def run_one(index, item):
            async with semaphore:
                try:
                    response = await self._client.complete_chat(
                        messages=item.messages,
                        model=item.model or settings.groq_default_model,
                        temperature=item.temperature or 0.7,
                    )

                    # 🔴 DB write YOK burada
                    return {
                        "index": index,
                        "status": "ok",
                        "response": response,
                        "messages": item.messages,
                        "model": item.model or settings.groq_default_model,
                        "temperature": item.temperature,
                    }

                except Exception as e:
                    return {
                        "index": index,
                        "status": "error",
                        "error": str(e),
                    }

        # 1️⃣ LLM çağrıları (paralel)
        llm_results = await asyncio.gather(
            *[run_one(i, item) for i, item in enumerate(items)]
        )

        # 2️⃣ DB write'lar (TEK TEK)
        for r in llm_results:
            if r["status"] == "ok":
                await create_chat_log(
                    session=session,
                    prompt=self._extract_user_prompt(r["messages"]),
                    messages=r["messages"],
                    response=r["response"],
                    model_name=r["model"],   # 🔥 NULL DEĞİL
                    temperature=r["temperature"],
                )

        return llm_results
