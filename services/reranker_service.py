import logging
from typing import Dict, List

from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Инициализация reranker модели.

        Args:
            model_name: Название модели reranker
                Рекомендуемые модели:
                - BAAI/bge-reranker-v2-m3 (мультиязычная, balanced)
                - BAAI/bge-reranker-large (английская, высокая точность)
                - BAAI/bge-reranker-base (быстрая, базовая)
        """
        logger.info(f"Initializing reranker model: {model_name}")
        try:
            self.model = FlagReranker(model_name, use_fp16=True)
            self.model_name = model_name
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model {model_name}: {e}")
            logger.info("Falling back to BAAI/bge-reranker-base")
            self.model = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
            self.model_name = "BAAI/bge-reranker-base"

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Переранжировать документы на основе релевантности к запросу.

        Args:
            query: Запрос пользователя
            documents: Список документов с полями 'text', 'score' и метаданными
            top_k: Количество документов для возврата после reranking

        Returns:
            Отсортированный список документов с обновленными scores
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []

        logger.info(
            f"Reranking {len(documents)} documents for query: '{query[:50]}...'"
        )

        # Подготовка пар (query, document) для reranking
        pairs = [[query, doc["text"]] for doc in documents]

        try:
            # Получаем scores от reranker
            scores = self.model.compute_score(pairs, normalize=True)

            # Если scores - одно число (один документ), делаем список
            if not isinstance(scores, list):
                scores = [scores]

            # Обновляем scores в документах
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
                doc["original_score"] = doc.get("score", 0.0)

            # Сортируем по rerank_score
            reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

            # Возвращаем top_k
            result = reranked[:top_k]

            logger.info(
                f"Reranking complete. Top score: {result[0]['rerank_score']:.4f}"
            )
            logger.info(
                f"Score range: {result[-1]['rerank_score']:.4f} - {result[0]['rerank_score']:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error during reranking: {e}", exc_info=True)
            # В случае ошибки возвращаем оригинальные документы
            return documents[:top_k]

    def rerank_with_threshold(
        self, query: str, documents: List[Dict], top_k: int = 5, threshold: float = 0.3
    ) -> List[Dict]:
        """
        Переранжировать с порогом релевантности.

        Args:
            query: Запрос пользователя
            documents: Список документов
            top_k: Максимальное количество документов
            threshold: Минимальный score для включения документа

        Returns:
            Отфильтрованный и отсортированный список документов
        """
        reranked = self.rerank(query, documents, top_k=len(documents))

        # Фильтруем по threshold
        filtered = [doc for doc in reranked if doc.get("rerank_score", 0) >= threshold]

        logger.info(
            f"Filtered {len(filtered)}/{len(reranked)} documents above threshold {threshold}"
        )

        return filtered[:top_k]

    def rerank_with_diversity(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
        diversity_weight: float = 0.3,
    ) -> List[Dict]:
        """
        Переранжировать с учетом разнообразия источников.

        Args:
            query: Запрос пользователя
            documents: Список документов
            top_k: Количество документов для возврата
            diversity_weight: Вес разнообразия (0-1)

        Returns:
            Список документов с балансом релевантности и разнообразия
        """
        if not documents:
            return []

        # Сначала обычный reranking
        reranked = self.rerank(query, documents, top_k=len(documents))

        # MMR (Maximal Marginal Relevance) для разнообразия
        selected = []
        seen_pages = set()

        for doc in reranked:
            if len(selected) >= top_k:
                break

            page_id = doc.get("page_id", "")

            # Бонус за новый источник
            diversity_bonus = 0.0
            if page_id and page_id not in seen_pages:
                diversity_bonus = diversity_weight
                seen_pages.add(page_id)

            # Финальный score
            doc["final_score"] = (
                doc["rerank_score"] * (1 - diversity_weight) + diversity_bonus
            )
            selected.append(doc)

        # Пересортируем по финальному score
        selected.sort(key=lambda x: x["final_score"], reverse=True)

        logger.info(
            f"Reranked with diversity: {len(seen_pages)} unique sources in top {len(selected)}"
        )

        return selected[:top_k]

    def batch_rerank(
        self, queries: List[str], documents_list: List[List[Dict]], top_k: int = 5
    ) -> List[List[Dict]]:
        """
        Batch reranking для нескольких запросов.

        Args:
            queries: Список запросов
            documents_list: Список списков документов (по одному на запрос)
            top_k: Количество документов для каждого запроса

        Returns:
            Список отсортированных списков документов
        """
        logger.info(f"Batch reranking for {len(queries)} queries")

        results = []
        for query, documents in zip(queries, documents_list):
            reranked = self.rerank(query, documents, top_k)
            results.append(reranked)

        return results
