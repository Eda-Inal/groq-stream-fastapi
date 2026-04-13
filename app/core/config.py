"""
Application configuration module.

Environment-based settings using pydantic.
"""

from pydantic_settings import BaseSettings
from typing import Dict, Any


# 1. Move AVAILABLE_MODELS OUTSIDE the class. 
# 2. This makes it a constant that can be imported directly.
AVAILABLE_MODELS: Dict[str, Any] = {
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

    tavily_api_key: str | None = None

    # ------------------------------------------------------------------
    # MCP
    # ------------------------------------------------------------------
    mcp_enabled: bool = False
    mcp_server_url: str | None = None
    mcp_timeout: float = 10.0

    # ------------------------------------------------------------------
    # LLM-as-a-Judge (PAIRWISE)
    # ------------------------------------------------------------------
    JUDGE_MODEL: str = "llama-3.3-70b-versatile"
    JUDGE_TEMPERATURE: float = 0.0
    JUDGE_RUNS: int = 1 
    EVAL_CANDIDATE_TEMPERATURE: float = 0.2

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str

    # ------------------------------------------------------------------
    # Embeddings (TODO-2)
    # ------------------------------------------------------------------
    embedding_model_name: str = "nomic-embed-text"
    embedding_dim: int = 768
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_timeout: float = 15.0
    embedding_max_retries: int = 3
    embedding_retry_backoff: float = 1.5
    embedding_cache_enabled: bool = True
    embedding_cache_max_entries: int = 1000

    # ------------------------------------------------------------------
    # RAG retrieval defaults (TODO-4)
    # ------------------------------------------------------------------
    rag_default_top_k: int = 5
    rag_similarity_threshold: float = 0.7
    rag_max_top_k: int = 20

    # ------------------------------------------------------------------
    # RAG / document chunking (TODO-1.5; used by chunk_document when args omitted)
    # ------------------------------------------------------------------
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 100
    max_document_tokens: int = 100_000
    short_doc_single_chunk_max_tokens: int = 100
    min_chunk_tokens: int = 20

    class Config:
        env_file = ".env"


settings = Settings()
