import logging
import time
from typing import List, Dict, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Set up a logger for this module
logger = logging.getLogger(__name__)

class QdrantService:
    """Сервис для работы с Qdrant."""

    def __init__(self, url: str, collection_name: str, vector_size: int):
        """
        Инициализация клиента Qdrant.

        Args:
            url: URL Qdrant сервера
            collection_name: Имя коллекции
            vector_size: Размерность векторов
        """
        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = None
        self._connect_with_retry()

    def _connect_with_retry(self, max_retries: int = 5, delay: int = 5):
        """
        Подключиться к Qdrant с повторными попытками.

        Args:
            max_retries: Максимальное количество попыток
            delay: Задержка между попытками в секундах
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to Qdrant at {self.url} ({attempt + 1}/{max_retries})...")
                self.client = QdrantClient(url=self.url)
                # Проверяем подключение
                self.client.get_collections()
                logger.info("Successfully connected to Qdrant.")
                return
            except Exception as e:
                logger.error(f"Error connecting to Qdrant: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.critical(f"Failed to connect to Qdrant after {max_retries} attempts.")
                    raise Exception(f"Не удалось подключиться к Qdrant после {max_retries} попыток")

    def init_collection(self):
        """Создать коллекцию если не существует."""
        logger.info(f"Initializing collection '{self.collection_name}'...")
        try:
            collections = self.client.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)

            if not exists:
                logger.info(f"Collection '{self.collection_name}' does not exist. Creating...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created successfully.")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Error initializing collection '{self.collection_name}': {e}", exc_info=True)
            raise

    def upsert_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict[str, str]
    ) -> int:
        """
        Загрузить чанки с эмбеддингами в Qdrant.

        Args:
            chunks: Список текстовых чанков
            embeddings: Список эмбеддингов
            metadata: Метаданные (title, url, page_id)

        Returns:
            Количество загруженных чанков
        """
        logger.info(f"Preparing to upsert {len(chunks)} chunks for page_id='{metadata.get('page_id')}'.")
        points = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    'text': chunk,
                    'title': metadata.get('title', ''),
                    'url': metadata.get('url', ''),
                    'page_id': metadata.get('page_id', ''),
                    'chunk_index': i
                }
            )
            points.append(point)

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully upserted {len(points)} points into collection '{self.collection_name}'.")
            return len(points)
        except Exception as e:
            logger.error(f"Failed to upsert chunks for page_id='{metadata.get('page_id')}': {e}", exc_info=True)
            raise # Re-raise the exception after logging

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        page_id_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Поиск похожих чанков.

        Args:
            query_embedding: Эмбеддинг запроса
            top_k: Количество результатов
            page_id_filter: Опциональный фильтр по page_id

        Returns:
            Список найденных чанков с метаданными и score
        """
        logger.info(f"Searching in collection '{self.collection_name}' for top_k={top_k} results.")
        search_filter = None
        if page_id_filter:
            logger.info(f"Applying filter for page_id: {page_id_filter}")
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="page_id",
                        match=MatchValue(value=page_id_filter)
                    )
                ]
            )

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=search_filter
            ).points

            logger.info(f"Found {len(results)} matching chunks.")
            return [
                {
                    'text': hit.payload.get('text', ''),
                    'title': hit.payload.get('title', ''),
                    'url': hit.payload.get('url', ''),
                    'page_id': hit.payload.get('page_id', ''),
                    'score': hit.score
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Search failed in collection '{self.collection_name}': {e}", exc_info=True)
            return [] # Return empty list on failure

    def delete_by_page_id(self, page_id: str) -> bool:
        """
        Удалить все чанки страницы.

        Args:
            page_id: ID страницы Confluence

        Returns:
            True если успешно
        """
        logger.info(f"Attempting to delete all chunks for page_id: {page_id}")
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="page_id",
                            match=MatchValue(value=page_id)
                        )
                    ]
                )
            )
            logger.info(f"Successfully deleted chunks for page_id: {page_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks for page_id '{page_id}': {e}", exc_info=True)
            return False

