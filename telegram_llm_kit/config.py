from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Required
    telegram_bot_token: str
    llm_api_key: str
    llm_provider: Literal["deepseek", "claude"]

    # Optional with defaults
    llm_model: str | None = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024
    embedding_model: str = "all-MiniLM-L6-v2"
    recency_count: int = 20
    semantic_count: int = 10
    sqlite_db_path: str = "data/bot.db"
    chroma_persist_dir: str = "data/chroma"
    log_level: str = "INFO"
