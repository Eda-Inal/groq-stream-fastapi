import itertools
import json
import re
from typing import Iterable
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.chat_log import ChatLog
from app.db.repositories.chat_pairwise_evaluation import ChatPairwiseEvaluationRepository
from app.evaluation.judge_client import JudgeClient
from app.evaluation.pairwise_prompt import PAIRWISE_JUDGE_PROMPT_V1
from app.evaluation.schemas import JudgePairwiseResult
from app.services.groq_client import GroqClient
from app.core.config import settings


def extract_json_object(text: str) -> dict:
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


class PairwiseEvaluationService:
    def __init__(self, *, judge_client: JudgeClient) -> None:
        self.judge_client = judge_client
        self._llm = GroqClient()

    async def _generate_answer(self, *, prompt: str, model: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return await self._llm.complete_chat(
            messages=messages,
            model=model,
            temperature=getattr(settings, "EVAL_CANDIDATE_TEMPERATURE", 0.2),
        )

    async def _judge_pair(
            self, 
            *, 
            user_prompt: str, 
            answer_a: str, 
            answer_b: str, 
            rubric_version: str,
            model_a: str, # Added this
            model_b: str  # Added this
        ) -> JudgePairwiseResult:
            
            prompt_1 = (
                PAIRWISE_JUDGE_PROMPT_V1
                .replace("{user_prompt}", str(user_prompt))
                .replace("{answer_a}", str(answer_a))
                .replace("{answer_b}", str(answer_b))
            )
            
            prompt_2 = (
                PAIRWISE_JUDGE_PROMPT_V1
                .replace("{user_prompt}", str(user_prompt))
                .replace("{answer_a}", str(answer_b)) 
                .replace("{answer_b}", str(answer_a)) 
            )

            raw_1, raw_2 = await asyncio.gather(
                self.judge_client.evaluate(prompt=prompt_1),
                self.judge_client.evaluate(prompt=prompt_2)
            )

            res_1 = JudgePairwiseResult.model_validate(extract_json_object(raw_1))
            res_2 = JudgePairwiseResult.model_validate(extract_json_object(raw_2))

            final_winner = "Inconsistent"
            
            if res_1.winner == "A" and res_2.winner == "B":
                final_winner = "A"
            elif res_1.winner == "B" and res_2.winner == "A":
                final_winner = "B"
            elif res_1.winner == "Tie" and res_2.winner == "Tie":
                final_winner = "Tie"
            else:
                final_winner = "Inconsistent"

            combined_notes = (
                f"Comparison: {model_a} (A) vs {model_b} (B)\n"
                f"[Round 1]: {res_1.notes}\n"
                f"[Round 2]: {res_2.notes}"
            )

            return JudgePairwiseResult(winner=final_winner, notes=combined_notes)

    async def evaluate_chat_log(
        self,
        *,
        session: AsyncSession,
        chat_log: ChatLog,
        candidate_models: list[str],
        rubric_version: str = "v1",
        all_pairs: bool = True,
    ) -> None:
        user_prompt = (chat_log.prompt or "").strip()
        if not user_prompt:
            return

        # 1) Generate answers for each candidate model in parallel
        tasks = [self._generate_answer(prompt=user_prompt, model=m) for m in candidate_models]
        results = await asyncio.gather(*tasks)
        answers = dict(zip(candidate_models, results))

        # 2) Decide which pairs to evaluate
        if all_pairs:
            pairs = list(itertools.combinations(candidate_models, 2))
        else:
            if len(candidate_models) != 2:
                raise ValueError("all_pairs=False requires exactly 2 candidate models")
            pairs = [(candidate_models[0], candidate_models[1])]

        # 3) Judge each pair and persist
        for model_a, model_b in pairs:
            try:
                answer_a = answers[model_a]
                answer_b = answers[model_b]

                result = await self._judge_pair(
                    user_prompt=user_prompt,
                    answer_a=answer_a,
                    answer_b=answer_b,
                    rubric_version=rubric_version,
                    model_a=model_a, # Pass the name here
                    model_b=model_b  # Pass the name here
                )

                await ChatPairwiseEvaluationRepository.upsert(
                    session=session,
                    chat_log_id=chat_log.id,
                    rubric_version=rubric_version,
                    candidate_model_a=model_a,
                    candidate_model_b=model_b,
                    answer_a=answer_a,
                    answer_b=answer_b,
                    judge_model_name=self.judge_client.model_name,
                    winner=result.winner,
                    notes=result.notes,
                )
            except Exception as e:
                print(f"Error judging pair {model_a} vs {model_b} for log {chat_log.id}: {e}")
                continue