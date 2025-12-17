import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        logger.info(f"[Embedding] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._setup_prefixes()
        logger.info(f"[Embedding] Model loaded: {model_name}")
        logger.info(
            f"[Embedding] Prefixes - Query: '{self.query_prefix}', Passage: '{self.passage_prefix}'"
        )

    def _setup_prefixes(self):
        model_lower = self.model_name.lower()

        if "bge" in model_lower:
            self.query_prefix = "Представьте запрос для поиска релевантных документов: "
            self.passage_prefix = ""
            self.use_prefixes = True
            logger.info("[Embedding] BGE model detected - using instruction prefixes")
        elif "e5" in model_lower:
            self.query_prefix = "query: "
            self.passage_prefix = "passage: "
            self.use_prefixes = True
            logger.info("[Embedding] E5 model detected - using query/passage prefixes")
        else:
            self.query_prefix = ""
            self.passage_prefix = ""
            self.use_prefixes = False
            logger.info("[Embedding] Generic model - no prefixes")

    def embed_query(self, query: str) -> List[float]:
        if self.use_prefixes:
            query = self.query_prefix + query

        logger.debug(f"[Embedding] Query embedding: {len(query)} chars")
        embedding = self.model.encode(
            query, convert_to_tensor=False, normalize_embeddings=True
        )
        logger.debug("[Embedding] Query embedding generated")
        return embedding.tolist()

    def embed_passage(self, text: str) -> List[float]:
        if self.use_prefixes:
            text = self.passage_prefix + text

        logger.debug(f"[Embedding] Passage embedding: {len(text)} chars")
        embedding = self.model.encode(
            text, convert_to_tensor=False, normalize_embeddings=True
        )
        logger.debug("[Embedding] Passage embedding generated")
        return embedding.tolist()

    def embed(self, text: str, is_query: bool = False) -> List[float]:
        return self.embed_query(text) if is_query else self.embed_passage(text)

    def embed_batch_passages(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"[Embedding] Batch passages: {len(texts)} items")

        if self.use_prefixes:
            texts = [self.passage_prefix + text for text in texts]

        embeddings = self.model.encode(
            texts,
            convert_to_tensor=False,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        logger.info("[Embedding] Batch passages complete")
        return [emb.tolist() for emb in embeddings]

    def embed_batch_queries(self, queries: List[str]) -> List[List[float]]:
        logger.info(f"[Embedding] Batch queries: {len(queries)} items")

        if self.use_prefixes:
            queries = [self.query_prefix + query for query in queries]

        embeddings = self.model.encode(
            queries,
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        logger.info("[Embedding] Batch queries complete")
        return [emb.tolist() for emb in embeddings]

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode_for_retrieval(self, text: str, is_query: bool = False) -> List[float]:
        return self.embed(text, is_query=is_query)
