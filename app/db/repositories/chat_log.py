from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chat_log import ChatLog


async def create_chat_log(
    session: AsyncSession,
    *,
    prompt: str,
    response: str,
    model_name: str,
    messages: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    seed: int | None = None,
) -> ChatLog:
    """
    Persist a single LLM interaction into the database.

    - OpenAI-compatible messages are stored as JSONB (source of truth)
    - LLM parameters are persisted for audit / replay
    - Single DB write (after stream completion)
    """
    chat_log = ChatLog(
        prompt=prompt,
        messages=messages,
        response=response,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
    )

    session.add(chat_log)
    await session.commit()
    await session.refresh(chat_log)

    return chat_log
