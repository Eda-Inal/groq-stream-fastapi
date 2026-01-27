import argparse
import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import engine
from app.db.models.chat_log import ChatLog
from app.evaluation.judge_client import JudgeClient
from app.evaluation.service import PairwiseEvaluationService
from app.core.config import settings


def _parse_models(value: str) -> list[str]:
    # comma-separated: "modelA,modelB,modelC"
    items = [x.strip() for x in value.split(",") if x.strip()]
    if len(items) < 2:
        raise ValueError("Provide at least 2 candidate models")
    return items


async def run(limit: int, candidate_models: list[str], all_pairs: bool) -> None:
    judge_client = JudgeClient()
    service = PairwiseEvaluationService(judge_client=judge_client)

    async with AsyncSession(engine) as session:
        stmt = select(ChatLog).order_by(ChatLog.created_at.desc()).limit(limit)
        result = await session.execute(stmt)
        chat_logs = result.scalars().all()

        if not chat_logs:
            print("No chat logs found.")
            return

        print(f"Evaluating {len(chat_logs)} chat logs (pairwise)...")
        print(f"Judge: {judge_client.model_name}")
        print(f"Candidates: {candidate_models}")
        print(f"All pairs: {all_pairs}")

        for chat_log in chat_logs:
            try:
                await service.evaluate_chat_log(candidate_models=candidate_models, all_pairs=all_pairs, session=session, chat_log=chat_log)
                print(f"✓ Evaluated chat_log_id={chat_log.id}")
            except Exception as exc:
                print(f"✗ Failed chat_log_id={chat_log.id}: {exc}")

        await session.commit()
        print("Pairwise evaluation run completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise LLM-as-a-Judge evaluation")
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help='Comma-separated candidate models, e.g. "llama-3.1-8b-instruct,mixtral-8x7b-instruct,gemma-7b-it"',
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="If set, evaluate all pairs (A–B, A–C, B–C...). Otherwise evaluates only the first two models (A vs B).",
    )

    args = parser.parse_args()
    candidate_models = _parse_models(args.candidates)
    asyncio.run(run(limit=args.limit, candidate_models=candidate_models, all_pairs=args.all_pairs))


if __name__ == "__main__":
    main()
