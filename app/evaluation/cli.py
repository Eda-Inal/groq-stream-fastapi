import argparse
import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import engine

from app.db.models.chat_log import ChatLog
from app.evaluation.judge_client import JudgeClient
from app.evaluation.service import EvaluationService


async def run(limit: int) -> None:
    judge_client = JudgeClient()
    service = EvaluationService(judge_client=judge_client)

    async with AsyncSession(engine) as session:
        stmt = (
            select(ChatLog)
            .order_by(ChatLog.created_at.desc())
            .limit(limit)
        )

        result = await session.execute(stmt)
        chat_logs = result.scalars().all()

        if not chat_logs:
            print("No chat logs found.")
            return

        print(f"Evaluating {len(chat_logs)} chat logs...")

        for chat_log in chat_logs:
            try:
                await service.evaluate_chat_log(
                    session=session,
                    chat_log=chat_log,
                )
                print(f"✓ Evaluated chat_log_id={chat_log.id}")
            except Exception as exc:
                print(f"✗ Failed chat_log_id={chat_log.id}: {exc}")

        await session.commit()
        print("Evaluation run completed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the last N chat logs using LLM-as-a-Judge"
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help="Number of recent chat logs to evaluate",
    )

    args = parser.parse_args()
    asyncio.run(run(limit=args.limit))


if __name__ == "__main__":
    main()
