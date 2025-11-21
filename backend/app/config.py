from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    confluence_url: str
    confluence_username: str
    confluence_token: str
    confluence_space_key: str
    log_level: str = "INFO"
    allowed_origins: list[str] = ["https://assistant.mfactory.nxcloud.nexign.com"]

    llm_url: str
    llm_api_key: str

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vector_db_mode: Literal["persistent", "memory"] = "persistent"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

settings = Settings()

import logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
