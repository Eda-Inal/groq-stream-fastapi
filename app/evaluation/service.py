import json
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chat_log import ChatLog
from app.db.repositories.chat_evaluation import ChatEvaluationRepository
from app.evaluation.judge_client import JudgeClient
from app.evaluation.prompt import JUDGE_PROMPT_V1
from app.evaluation.schemas import JudgeEvaluationResult


def extract_json_object(text: str) -> dict:
    """
    Extract and parse a JSON object from judge model output.
    """

    if not text:
        raise ValueError("Empty judge output")

    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    if (
        (text.startswith("'") and text.endswith("'"))
        or (text.startswith('"') and text.endswith('"'))
    ):
        text = text[1:-1].strip()

    match = re.search(r"\{[\s\S]*?\}", text)
    if not match:
        raise ValueError(f"No JSON object found in judge output:\n{text}")

    return json.loads(match.group(0))


class EvaluationService:
    """
    Orchestrates LLM-as-a-Judge evaluation and persistence.
    """

    def __init__(self, *, judge_client: JudgeClient) -> None:
        self.judge_client = judge_client

    async def evaluate_chat_log(
        self,
        *,
        session: AsyncSession,
        chat_log: ChatLog,
        rubric_version: str = "v1",
    ) -> None:
        """
        Evaluate a single ChatLog and persist the evaluation result.
        """

        user_prompt = chat_log.prompt or ""
        model_response = chat_log.response or ""

        prompt = (
            JUDGE_PROMPT_V1
            .replace("{user_prompt}", str(user_prompt))
            .replace("{model_response}", str(model_response))
        )

        raw_output = await self.judge_client.evaluate(prompt=prompt)

        parsed = extract_json_object(raw_output)
        result = JudgeEvaluationResult.model_validate(parsed)

        await ChatEvaluationRepository.upsert(
            session=session,
            chat_log_id=chat_log.id,
            judge_model_name=self.judge_client.model_name,
            rubric_version=rubric_version,
            relevance=result.relevance,
            completeness=result.completeness,
            clarity=result.clarity,
            overall_score=result.overall_score,
            notes=result.notes,
        )
