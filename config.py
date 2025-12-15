import logging
from functools import lru_cache

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    confluence_url: str
    confluence_username: str
    confluence_api_token: str

    llm_base_url: str
    llm_api_key: str
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "confluence"

    embedding_model: str = "ai-forever/sbert_large_nlu_ru"
    embedding_dimension: int = 1024

    chunk_size: int = 500
    chunk_overlap: int = 50

    top_k_results: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    logger.info("Loading application settings...")
    settings = Settings()
    logger.info("Application settings loaded successfully.")
    return settings
