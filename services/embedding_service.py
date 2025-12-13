from typing import List
import logging
from sentence_transformers import SentenceTransformer
import re

# Set up a logger for this module
logger = logging.getLogger(__name__)

class EmbeddingService:
    """Сервис для генерации эмбеддингов текста."""
    
    def __init__(self, model_name: str):
        """
        Инициализация модели эмбеддингов.
        
        Args:
            model_name: Название модели sentence-transformers
        """
        logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")
    def embed(self, text: str) -> List[float]:
        """
        Получить эмбеддинг для текста.
        
        Args:
            text: Текст для эмбеддинга
            
        Returns:
            Вектор эмбеддинга
        """
        logger.debug(f"Generating embedding for text of length {len(text)} characters...")
        embedding = self.model.encode(text, convert_to_tensor=False)
        logger.debug("Embedding generated successfully")
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Получить эмбеддинги для батча текстов.
        
        Args:
            texts: Список текстов
            
        Returns:
            Список векторов эмбеддингов
        """
        logger.info(f"Generating embeddings for batch of {len(texts)} texts...")
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        logger.info("Embeddings generated successfully")
        return [emb.tolist() for emb in embeddings]
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Разбить текст на чанки по словам с перекрытием.
        
        Args:
            text: Исходный текст
            chunk_size: Размер чанка в словах
            overlap: Перекрытие между чанками в словах
            
        Returns:
            Список чанков
        """
        logger.info(f"Chunking text of length {len(text)} characters...")
        # Разбиваем на слова
        words = re.findall(r'\S+', text)
        
        if len(words) <= chunk_size:
            logger.debug(f"Text is shorter than chunk size ({len(words)} <= {chunk_size}). Returning single chunk.")
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            
            # Двигаемся с учетом overlap
            start += chunk_size - overlap
            
            # Если остался маленький кусок, добавляем его и завершаем
            if end >= len(words):
                break
        
        logger.info(f"Successfully created {len(chunks)} chunks")
        return chunks