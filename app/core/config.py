"""
Application configuration module.

Environment-based settings using pydantic.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # ------------------------------------------------------------------
    # Groq API
    # ------------------------------------------------------------------
    groq_api_key: str
    groq_base_url: str
    groq_default_model: str

    groq_verify_ssl: bool = True
    groq_connect_timeout: float = 10.0
    groq_read_timeout: float = 30.0

    # ------------------------------------------------------------------
    # LLM-as-a-Judge (PAIRWISE)
    # ------------------------------------------------------------------

    # Fixed judge model (NEVER a candidate)
    JUDGE_MODEL: str = "llama-3.3-70b-versatile"

    # Deterministic judging
    JUDGE_TEMPERATURE: float = 0.0

    # How many times to run the judge per pair (majority vote ready)
    JUDGE_RUNS: int = 1  # set to 3 later if needed

    # ------------------------------------------------------------------
    # Candidate generation (models being compared)
    # ------------------------------------------------------------------

    # Temperature for candidate answers (slightly > 0 is fine)
    EVAL_CANDIDATE_TEMPERATURE: float = 0.2

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str

    class Config:
        env_file = ".env"


# ----------------------------------------------------------------------
# OpenAI-compatible model registry (server-side whitelist)
# ----------------------------------------------------------------------

AVAILABLE_MODELS = {
    # ----------------------
    # Judge (not selectable by users)
    # ----------------------
    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "tier": "judge",
        "stream": False,
    },

    # ----------------------
    # Candidate models
    # ----------------------
    "llama-3.1-8b-instant": {
        "provider": "groq",
        "tier": "fast",
        "stream": True,
    },
        "meta-llama/llama-4-maverick-17b-128e-instruct": {
        "provider": "groq",
        "tier": "balanced",
        "stream": True,
    },


}

settings = Settings()
