from app.core.config import settings
from app.services.groq_client import GroqClient


class JudgeClient:
    """
    Client responsible for evaluating model responses
    using an LLM-as-a-Judge approach.
    """

    def __init__(self) -> None:
        self.model_name = settings.JUDGE_MODEL
        self._client = GroqClient()

    async def evaluate(self, *, prompt: str) -> str:
        """
        Evaluate a prompt-response pair using a non-streaming LLM call.

        Returns the raw evaluation output produced by the judge model.
        """

        messages = [
            {
                "role": "system",
                "content": "You are a strict evaluator. Return ONLY the evaluation output.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        return await self._client.complete_chat(
            messages=messages,
            model=self.model_name,
            temperature=settings.JUDGE_TEMPERATURE,
        )
