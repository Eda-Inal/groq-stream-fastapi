from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chat_log import ChatLog


async def create_chat_log(
    session: AsyncSession,
    *,
    prompt: str,
    response: str,
    model_name: str,
) -> ChatLog:
    """
    Persist a single LLM interaction into the database.

    This function is intentionally isolated from
    API / streaming / LLM logic.
    """
    chat_log = ChatLog(
        prompt=prompt,
        response=response,
        model_name=model_name,
    )

    session.add(chat_log)
    await session.commit()
    await session.refresh(chat_log)

    return chat_log
