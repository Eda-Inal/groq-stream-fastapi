from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Attributes:
        groq_api_key: API key for Groq service.
        groq_base_url: Base URL for Groq OpenAI-compatible API.
        groq_default_model: Default LLM model name.
    """

    groq_api_key: str
    groq_base_url: str
    groq_default_model: str

    class Config:
        env_file = ".env"


settings = Settings()
