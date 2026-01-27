from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chat_pairwise_evaluation import ChatPairwiseEvaluation


class ChatPairwiseEvaluationRepository:
    @staticmethod
    async def upsert(
        *,
        session: AsyncSession,
        chat_log_id: int,
        rubric_version: str,
        candidate_model_a: str,
        candidate_model_b: str,
        answer_a: str,
        answer_b: str,
        judge_model_name: str,
        winner: str,
        notes: str,
    ) -> ChatPairwiseEvaluation:
        stmt = select(ChatPairwiseEvaluation).where(
            ChatPairwiseEvaluation.chat_log_id == chat_log_id,
            ChatPairwiseEvaluation.rubric_version == rubric_version,
            ChatPairwiseEvaluation.candidate_model_a == candidate_model_a,
            ChatPairwiseEvaluation.candidate_model_b == candidate_model_b,
        )

        result = await session.execute(stmt)
        row = result.scalar_one_or_none()

        if row is None:
            row = ChatPairwiseEvaluation(
                chat_log_id=chat_log_id,
                rubric_version=rubric_version,
                candidate_model_a=candidate_model_a,
                candidate_model_b=candidate_model_b,
                answer_a=answer_a,
                answer_b=answer_b,
                judge_model_name=judge_model_name,
                winner=winner,
                notes=notes,
            )
            session.add(row)
        else:
            row.answer_a = answer_a
            row.answer_b = answer_b
            row.judge_model_name = judge_model_name
            row.winner = winner
            row.notes = notes

        await session.flush()
        return row
