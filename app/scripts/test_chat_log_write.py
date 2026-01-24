import asyncio

from app.db.session import AsyncSessionLocal
from app.db.repositories.chat_log import create_chat_log


async def main() -> None:
    async with AsyncSessionLocal() as session:
        chat_log = await create_chat_log(
            session,
            prompt="Hello LLM",
            response="Hello human",
            model_name="test-model",
        )

        print("Inserted chat_log id:", chat_log.id)


if __name__ == "__main__":
    asyncio.run(main())
