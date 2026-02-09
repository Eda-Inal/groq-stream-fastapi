from typing import Any, List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

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
    conversation_id: str | None = None,
    turn_index: int | None = None,
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
        conversation_id=conversation_id,
        turn_index=turn_index,
    )

    session.add(chat_log)
    await session.flush()
    await session.refresh(chat_log)

    return chat_log


async def list_chat_logs_by_conversation(
    session: AsyncSession,
    conversation_id: str,
    limit: int = 20,
) -> List[ChatLog]:
    """
    Fetch recent chat logs for a given conversation_id.

    Returns at most `limit` items, ordered from oldest to newest.
    Priority ordering:
      1) turn_index (if present)
      2) created_at (fallback)
    """
    stmt = (
        select(ChatLog)
        .where(ChatLog.conversation_id == conversation_id)
        .order_by(ChatLog.turn_index.asc().nulls_last(), ChatLog.created_at.asc())
        .limit(limit)
    )

    result = await session.execute(stmt)
    return list(result.scalars().all())
