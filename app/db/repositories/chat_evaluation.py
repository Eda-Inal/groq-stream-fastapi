from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chat_evaluation import ChatEvaluation


class ChatEvaluationRepository:
    @staticmethod
    async def upsert(
        *,
        session: AsyncSession,
        chat_log_id: int,
        judge_model_name: str,
        rubric_version: str,
        relevance: int,
        completeness: int,
        clarity: int,
        overall_score: int,
        notes: str,
    ) -> ChatEvaluation:
        """
        Insert or update an evaluation for a given chat_log + rubric_version.
        """

        stmt = select(ChatEvaluation).where(
            ChatEvaluation.chat_log_id == chat_log_id,
            ChatEvaluation.rubric_version == rubric_version,
        )

        result = await session.execute(stmt)
        evaluation = result.scalar_one_or_none()

        if evaluation is None:
            evaluation = ChatEvaluation(
                chat_log_id=chat_log_id,
                judge_model_name=judge_model_name,
                rubric_version=rubric_version,
                relevance=relevance,
                completeness=completeness,
                clarity=clarity,
                overall_score=overall_score,
                notes=notes,
            )
            session.add(evaluation)
        else:
            evaluation.judge_model_name = judge_model_name
            evaluation.relevance = relevance
            evaluation.completeness = completeness
            evaluation.clarity = clarity
            evaluation.overall_score = overall_score
            evaluation.notes = notes

        await session.flush()
        return evaluation
