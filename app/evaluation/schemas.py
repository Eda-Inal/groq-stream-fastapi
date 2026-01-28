from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class JudgePairwiseResult(BaseModel):
    winner: Literal["A", "B", "Tie", "Inconsistent"]
    notes: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")
