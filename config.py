import logging
from functools import lru_cache

from pydantic_settings import BaseSettings

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
    embedding_model: str = "BAAI/bge-m3"  # Обновлено на BGE-M3
    embedding_dimension: int = 1024  # BGE-M3 размерность

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_reranker: bool = True
    reranker_top_k: int = 5
    reranker_threshold: float = 0.3

    # Chunking (теперь в токенах)
    chunk_size: int = 512  # токенов
    chunk_overlap: int = 50  # токенов
    chunking_strategy: str = "semantic"  # "semantic" или "fixed"
    min_chunk_size: int = 100  # минимальный размер чанка в токенах

    # Tokenizer
    tokenizer_model: str = "cl100k_base"  # tiktoken model
    tiktoken_local_path: str = "/tmp/tiktoken"

    # RAG
    top_k_results: int = 20  # Больше для reranking
    final_top_k: int = 5  # После reranking
    use_hyde: bool = True  # Использовать HyDE
    use_diversity: bool = True  # Разнообразие источников
    diversity_weight: float = 0.3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Получить синглтон настроек."""
    logger.info("Loading application settings...")
    settings = Settings()
    logger.info("Application settings loaded successfully.")
    logger.info(f"Embedding model: {settings.embedding_model}")
    logger.info(
        f"Reranker model: {settings.reranker_model} (enabled: {settings.use_reranker})"
    )
    logger.info(
        f"Chunking: {settings.chunking_strategy} strategy, {settings.chunk_size} tokens"
    )
    logger.info(f"HyDE: {'enabled' if settings.use_hyde else 'disabled'}")
    return settings
