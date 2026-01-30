from collections.abc import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.repositories.chat_log import create_chat_log
from app.services.groq_client import GroqClient
from app.services.tools.registry import ToolRegistry
import asyncio

class ChatService:
    def __init__(self) -> None:
        self._client = GroqClient()
        self._tools = ToolRegistry()

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
        
        # 1. Apply math tools / pre-processing
        processed_messages = await self._tools.maybe_apply(messages)

        try:
            # 2. Stream from Groq (Passing all parameters)
            async for event in self._client.stream_chat_completion(
                messages=processed_messages,
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

            # 3. DB write happens here after stream finishes successfully
            if full_response:
                await create_chat_log(
                    session=session,
                    prompt=self._extract_user_prompt(messages),
                    messages=processed_messages,
                    response="".join(full_response),
                    model_name=model or "default",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    seed=seed,
                )
                await session.commit()

        except Exception as e:
            await session.rollback()
            
            raise e

    @staticmethod
    def _extract_user_prompt(messages: list[dict[str, str]]) -> str:
        for message in messages:
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    async def bulk_complete(self, *, session, items, concurrency: int = 5):
        from asyncio import Semaphore
        from app.core.config import settings

        semaphore = Semaphore(concurrency)

        async def run_one(index, item):
            async with semaphore:
                try:
                    response = await self._client.complete_chat(
                        messages=item.messages,
                        model=item.model or settings.groq_default_model,
                        temperature=item.temperature or 0.7,
                    )
                    return {
                        "index": index,
                        "status": "ok",
                        "response": response,
                        "messages": item.messages,
                        "model": item.model or settings.groq_default_model,
                        "temperature": item.temperature,
                    }
                except Exception as e:
                    return {"index": index, "status": "error", "error": str(e)}

        llm_results = await asyncio.gather(*[run_one(i, item) for i, item in enumerate(items)])

        for r in llm_results:
            if r["status"] == "ok":
                await create_chat_log(
                    session=session,
                    prompt=self._extract_user_prompt(r["messages"]),
                    messages=r["messages"],
                    response=r["response"],
                    model_name=r["model"],
                    temperature=r["temperature"],
                )
        
        await session.commit()
        return llm_results