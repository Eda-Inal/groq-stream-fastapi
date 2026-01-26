from pydantic import BaseModel, Field, ConfigDict


class JudgeEvaluationResult(BaseModel):
    relevance: int = Field(..., ge=1, le=5)
    completeness: int = Field(..., ge=1, le=5)
    clarity: int = Field(..., ge=1, le=5)
    overall_score: int = Field(..., ge=1, le=5)
    notes: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")
