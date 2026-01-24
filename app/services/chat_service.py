from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.chat_log import create_chat_log
from app.services.groq_client import GroqClient


class ChatService:
    """
    Orchestrates LLM streaming and persistence.

    - Streams chunks to the caller
    - Accumulates final response
    - Persists prompt/response after stream completion
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
            if event["type"] == "chunk":
                full_response.append(event["text"])

            if event["type"] == "done":
                final_response = "".join(full_response)

                # Persist once, after streaming is finished
                await create_chat_log(
                    session=session,
                    prompt=self._extract_prompt(messages),
                    response=final_response,
                    model_name=model or "default",
                )

            yield event

    @staticmethod
    def _extract_prompt(messages: list[dict[str, str]]) -> str:
        """
        Simple prompt extraction strategy.
        Can be improved later (system/user separation, etc.).
        """
        return "\n".join(
            f'{m["role"]}: {m["content"]}'
            for m in messages
            if "role" in m and "content" in m
        )
