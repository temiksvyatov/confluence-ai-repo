import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Инициализация модели эмбеддингов с поддержкой префиксов.

        Args:
            model_name: Название модели sentence-transformers
                Рекомендуемые модели:
                - BAAI/bge-m3 (мультиязычная, SOTA)
                - BAAI/bge-large-en-v1.5 (английская, высокая точность)
                - intfloat/multilingual-e5-large (мультиязычная E5)
        """
        logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        # Определяем тип модели и нужные префиксы
        self._setup_prefixes()

        logger.info(f"Embedding model loaded successfully: {model_name}")
        logger.info(
            f"Using prefixes - Query: '{self.query_prefix}', Passage: '{self.passage_prefix}'"
        )

    def _setup_prefixes(self):
        """Настроить префиксы в зависимости от типа модели."""
        model_lower = self.model_name.lower()

        if "bge" in model_lower:
            # BGE модели требуют префиксы для оптимальной работы
            self.query_prefix = "Представьте запрос для поиска релевантных документов: "
            self.passage_prefix = ""
            self.use_prefixes = True
            logger.info("Detected BGE model - using instruction prefixes")

        elif "e5" in model_lower:
            # E5 модели также используют префиксы
            self.query_prefix = "query: "
            self.passage_prefix = "passage: "
            self.use_prefixes = True
            logger.info("Detected E5 model - using query/passage prefixes")

        else:
            # Другие модели не требуют префиксов
            self.query_prefix = ""
            self.passage_prefix = ""
            self.use_prefixes = False
            logger.info("Generic model detected - no prefixes used")

    def embed_query(self, query: str) -> List[float]:
        """
        Получить эмбеддинг для запроса.

        Args:
            query: Запрос пользователя

        Returns:
            Вектор эмбеддинга
        """
        if self.use_prefixes:
            query = self.query_prefix + query

        logger.debug(
            f"Generating query embedding for text of length {len(query)} characters..."
        )
        embedding = self.model.encode(
            query, convert_to_tensor=False, normalize_embeddings=True
        )
        logger.debug("Query embedding generated successfully")
        return embedding.tolist()

    def embed_passage(self, text: str) -> List[float]:
        """
        Получить эмбеддинг для документа/отрывка.

        Args:
            text: Текст документа

        Returns:
            Вектор эмбеддинга
        """
        if self.use_prefixes:
            text = self.passage_prefix + text

        logger.debug(
            f"Generating passage embedding for text of length {len(text)} characters..."
        )
        embedding = self.model.encode(
            text, convert_to_tensor=False, normalize_embeddings=True
        )
        logger.debug("Passage embedding generated successfully")
        return embedding.tolist()

    def embed(self, text: str, is_query: bool = False) -> List[float]:
        """
        Получить эмбеддинг для текста с автоматическим определением типа.

        Args:
            text: Текст для эмбеддинга
            is_query: True если это запрос, False если документ

        Returns:
            Вектор эмбеддинга
        """
        if is_query:
            return self.embed_query(text)
        else:
            return self.embed_passage(text)

    def embed_batch_passages(self, texts: List[str]) -> List[List[float]]:
        """
        Получить эмбеддинги для батча документов.

        Args:
            texts: Список текстов документов

        Returns:
            Список векторов эмбеддингов
        """
        logger.info(f"Generating embeddings for batch of {len(texts)} passages...")

        if self.use_prefixes:
            texts = [self.passage_prefix + text for text in texts]

        embeddings = self.model.encode(
            texts,
            convert_to_tensor=False,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        logger.info("Batch embeddings generated successfully")
        return [emb.tolist() for emb in embeddings]

    def embed_batch_queries(self, queries: List[str]) -> List[List[float]]:
        """
        Получить эмбеддинги для батча запросов.

        Args:
            queries: Список запросов

        Returns:
            Список векторов эмбеддингов
        """
        logger.info(f"Generating embeddings for batch of {len(queries)} queries...")

        if self.use_prefixes:
            queries = [self.query_prefix + query for query in queries]

        embeddings = self.model.encode(
            queries,
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        logger.info("Batch query embeddings generated successfully")
        return [emb.tolist() for emb in embeddings]

    def get_embedding_dimension(self) -> int:
        """
        Получить размерность эмбеддингов модели.

        Returns:
            Размерность векторов
        """
        return self.model.get_sentence_embedding_dimension()

    def encode_for_retrieval(self, text: str, is_query: bool = False) -> List[float]:
        """
        Унифицированный метод кодирования для retrieval.
        Алиас для embed() для обратной совместимости.

        Args:
            text: Текст для кодирования
            is_query: True для запроса, False для документа

        Returns:
            Вектор эмбеддинга
        """
        return self.embed(text, is_query=is_query)
