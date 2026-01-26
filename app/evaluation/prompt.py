JUDGE_PROMPT_V1 = """
You must respond with a single JSON object.
Do not include markdown, explanations, or any extra text.
Your output must start with '{' and end with '}'.

You are an impartial evaluator.
Your task is to evaluate the quality of a model-generated response to a user prompt.

Evaluation Criteria (1–5):

- Relevance: Does the response directly address the user's request?
- Completeness: Does the response cover the key points implied by the prompt?
- Clarity: Is the response easy to understand and well structured?

Overall Score (1–5):
Reflect the general quality of the response.
If any criterion is very low, the overall score should not be high.

Required Output Format (JSON ONLY):

{
  "relevance": 1,
  "completeness": 1,
  "clarity": 1,
  "overall_score": 1,
  "notes": "Brief explanation (1–2 sentences)"
}

User Prompt:
{user_prompt}

Model Response:
{model_response}
""".strip()
