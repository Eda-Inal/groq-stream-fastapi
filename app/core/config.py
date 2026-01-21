"""
Application configuration module.

This module defines environment-based settings using pydantic,
allowing configuration via a `.env` file.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Attributes:
        groq_api_key: API key used to authenticate with Groq.
        groq_base_url: Base URL for the Groq OpenAI-compatible API.
        groq_default_model: Default LLM model name used for requests.
    """

    groq_api_key: str
    groq_base_url: str
    groq_default_model: str

    class Config:
        env_file = ".env"


settings = Settings()
