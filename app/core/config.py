"""
Application configuration module.

Environment-based settings using pydantic.
"""

from pydantic_settings import BaseSettings
from typing import Dict, Any


AVAILABLE_MODELS: Dict[str, Any] = {
    # ──────────────────────────────────────────────────────────────────
    # Groq
    # ──────────────────────────────────────────────────────────────────
    # Judge — internal, not user-selectable
    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "tier": "judge",
        "stream": False,
        "context_window": 131072,
        "rate_limits": {"rpm": 30, "rpd": 1000, "tpm": 12000, "tpd": 100000},
    },
    "llama-3.1-8b-instant": {
        "provider": "groq",
        "tier": "fast",
        "stream": False,
        "context_window": 131072,
        "rate_limits": {"rpm": 30, "rpd": 14400, "tpm": 6000, "tpd": 500000},
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "provider": "groq",
        "tier": "balanced",
        "stream": False,
        "context_window": 131072,
        "rate_limits": {"rpm": 30, "rpd": 1000, "tpm": 30000, "tpd": 500000},
    },
    "qwen/qwen3-32b": {
        "provider": "groq",
        "tier": "balanced",
        "stream": False,
        "context_window": 131072,
        "rate_limits": {"rpm": 60, "rpd": 1000, "tpm": 6000, "tpd": 500000},
    },
    "openai/gpt-oss-120b": {
        "provider": "groq",
        "tier": "balanced",
        "stream": False,
        "context_window": 131072,
        "rate_limits": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    },

    # ──────────────────────────────────────────────────────────────────
    # SambaNova
    # ──────────────────────────────────────────────────────────────────
    "Meta-Llama-3.3-70B-Instruct": {
        "provider": "sambanova",
        "tier": "fast",
        "stream": False,
        "context_window": 131072,
    },

    # ──────────────────────────────────────────────────────────────────
    # OpenRouter
    # ──────────────────────────────────────────────────────────────────
    "openai/gpt-oss-120b:free": {
        "provider": "openrouter",
        "tier": "balanced",
        "stream": False,
        "context_window": 131072,
    },
    "google/gemma-4-31b-it:free": {
        "provider": "openrouter",
        "tier": "balanced",
        "stream": False,
        "context_window": 262144,
    },
    "google/gemma-4-26b-a4b-it:free": {
        "provider": "openrouter",
        "tier": "fast",
        "stream": False,
        "context_window": 262144,
    },
    "meta-llama/llama-3.3-70b-instruct:free": {
        "provider": "openrouter",
        "tier": "balanced",
        "stream": False,
        "context_window": 131072,
    },

    # ──────────────────────────────────────────────────────────────────
    # Google Gemini (Direct API — OpenAI-compatible)
    # Model IDs: https://ai.google.dev/gemini-api/docs/models
    # Disabled below: free-tier quota limit 0 (429) until billing is enabled.
    # ──────────────────────────────────────────────────────────────────
    # "gemini-2.5-pro": {
    #     "provider": "gemini",
    #     "tier": "balanced",
    #     "stream": False,
    #     "context_window": 1048576,
    # },
    "gemini-2.5-flash": {
        "provider": "gemini",
        "tier": "fast",
        "stream": False,
        "context_window": 1048576,
    },
    "gemini-3.5-flash": {
        "provider": "gemini",
        "tier": "balanced",
        "stream": False,
        "context_window": 1048576,
    },
    "gemini-2.5-flash-lite": {
        "provider": "gemini",
        "tier": "fast",
        "stream": False,
        "context_window": 1048576,
    },
    # "gemini-3.1-pro-preview": {
    #     "provider": "gemini",
    #     "tier": "balanced",
    #     "stream": False,
    #     "context_window": 1048576,
    # },
    # "gemini-2.5-computer-use-preview-10-2025": {
    #     "provider": "gemini",
    #     "tier": "preview",
    #     "stream": False,
    #     "context_window": 128000,
    # },

}


FALLBACK_CHAIN: list[str] = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-120b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
]


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # ------------------------------------------------------------------
    # Groq API (optional when using other providers)
    # ------------------------------------------------------------------
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_default_model: str = "llama-3.3-70b-versatile"

    groq_verify_ssl: bool = True
    groq_connect_timeout: float = 10.0
    groq_read_timeout: float = 30.0

    # ------------------------------------------------------------------
    # OpenRouter API (optional)
    # ------------------------------------------------------------------
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ------------------------------------------------------------------
    # Google Gemini Direct API (optional)
    # ------------------------------------------------------------------
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"

    # ------------------------------------------------------------------
    # SambaNova API (optional)
    # ------------------------------------------------------------------
    sambanova_api_key: str | None = None
    sambanova_base_url: str = "https://api.sambanova.ai/v1"

    tavily_api_key: str | None = None

    # ------------------------------------------------------------------
    # Tool Server
    # ------------------------------------------------------------------
    tool_server_url: str | None = None
    # Must exceed embedding_timeout * embedding_max_retries to avoid premature timeouts.
    tool_server_timeout: float = 120.0

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
    # Embeddings
    # ------------------------------------------------------------------
    embedding_model_name: str = "nomic-embed-text"
    embedding_dim: int = 768
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_timeout: float = 15.0
    embedding_max_retries: int = 3
    embedding_retry_backoff: float = 1.5
    embedding_batch_size: int = 50
    embedding_cache_enabled: bool = True
    embedding_cache_max_entries: int = 1000

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Tool availability flags
    # ------------------------------------------------------------------
    web_search_enabled: bool = True
    calculator_enabled: bool = True

    # ------------------------------------------------------------------
    # RAG retrieval defaults
    # ------------------------------------------------------------------
    rag_default_top_k: int = 5
    rag_similarity_threshold: float = 0.7
    rag_max_top_k: int = 20
    rag_system_prompt_enabled: bool = True
    max_context_tokens: int = 6000
    response_reserve_tokens: int = 1000
    rag_tool_max_context_tokens: int = 2500

    # ------------------------------------------------------------------
    # Hybrid search (dense + sparse BM25/FTS via PostgreSQL tsvector)
    # ------------------------------------------------------------------
    hybrid_search_enabled: bool = True
    # RRF constant — higher = less weight on top-ranked results (standard: 60)
    hybrid_rrf_k: int = 60
    # Candidates fetched per leg before RRF merging (multiplied by top_k)
    hybrid_fetch_multiplier: int = 3

    # ------------------------------------------------------------------
    # Grep search (third RRF leg — exact substring match via pg_trgm)
    # Catches structured identifiers (SVC-X-0000, ERR::CODE, dlq://...)
    # that tsvector tokenisation fragments. Requires pg_trgm extension
    # and GIN index on document_chunks.text (migration j5k6l7m8n9o0).
    # ------------------------------------------------------------------
    grep_search_enabled: bool = True

    # ------------------------------------------------------------------
    # Reranking (optional — improves retrieval precision)
    # ------------------------------------------------------------------
    reranker_enabled: bool = False
    reranker_base_url: str = "https://api.jina.ai/v1"
    reranker_api_key: str | None = None
    reranker_model: str = "jina-reranker-v2-base-multilingual"
    reranker_timeout: float = 10.0
    reranker_top_n: int | None = None
    reranker_overfetch_multiplier: int = 3

    # ------------------------------------------------------------------
    # RAG / document chunking (used by chunk_document when args omitted)
    # ------------------------------------------------------------------
    chunk_size_tokens: int = 300
    chunk_overlap_tokens: int = 75
    max_document_tokens: int = 100_000
    short_doc_single_chunk_max_tokens: int = 100
    min_chunk_tokens: int = 20

    # ------------------------------------------------------------------
    # Guardrails — small classifier LLM calls that run before the main
    # routing/tool/finalization pipeline to flag adversarial user input.
    # ------------------------------------------------------------------
    prompt_injection_guard_enabled: bool = True
    guard_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    guard_max_tokens: int = 8

    # ------------------------------------------------------------------
    # LangSmith tracing
    # ------------------------------------------------------------------
    langsmith_api_key: str | None = None
    langsmith_project: str = "groq-stream-fastapi"
    langsmith_tracing_enabled: bool = False

    class Config:
        env_file = ".env"


settings = Settings()
