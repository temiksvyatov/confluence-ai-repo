from pydantic_settings import BaseSettings
from functools import lru_cache
import logging

# Set up a logger for this module
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Настройки приложения из переменных окружения."""
    
    # Confluence
    confluence_url: str
    confluence_username: str
    confluence_api_token: str
    
    # LLM (OpenAI-compatible)
    llm_base_url: str
    llm_api_key: str
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    
    # Qdrant
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "confluence"
    
    # Embeddings
    embedding_model: str = "ai-forever/sbert_large_nlu_ru"
    embedding_dimension: int = 1024  # для sbert_large_nlu_ru
    
    # Chunking
    chunk_size: int = 500  # слов
    chunk_overlap: int = 50  # слов
    
    # RAG
    top_k_results: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings() -> Settings:
    """Получить синглтон настроек."""
    logger.info("Loading application settings...")
    settings = Settings()
    logger.info("Application settings loaded successfully.")
    return settings
