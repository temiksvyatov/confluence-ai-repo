import logging
from typing import Dict, List

from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        logger.info(f"[Reranker] Initializing model: {model_name}")
        try:
            self.model = FlagReranker(model_name, use_fp16=True)
            self.model_name = model_name
            logger.info("[Reranker] Model loaded successfully")
        except Exception as e:
            logger.error(f"[Reranker] Failed to load {model_name}: {e}")
            logger.info("[Reranker] Fallback to base model")
            self.model = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
            self.model_name = "BAAI/bge-reranker-base"

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        if not documents:
            logger.warning("[Reranker] No documents to rerank")
            return []

        logger.info(f"[Reranker] Reranking {len(documents)} documents")

        pairs = [[query, doc["text"]] for doc in documents]

        try:
            scores = self.model.compute_score(pairs, normalize=True)

            if not isinstance(scores, list):
                scores = [scores]

            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
                doc["original_score"] = doc.get("score", 0.0)

            reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
            result = reranked[:top_k]

            logger.info(
                f"[Reranker] Complete. Top score: {result[0]['rerank_score']:.4f}, "
                f"Range: {result[-1]['rerank_score']:.4f} - {result[0]['rerank_score']:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"[Reranker] Error: {e}", exc_info=True)
            return documents[:top_k]

    def rerank_with_threshold(
        self, query: str, documents: List[Dict], top_k: int = 5, threshold: float = 0.3
    ) -> List[Dict]:
        reranked = self.rerank(query, documents, top_k=len(documents))
        filtered = [doc for doc in reranked if doc.get("rerank_score", 0) >= threshold]

        logger.info(
            f"[Reranker] Filtered {len(filtered)}/{len(reranked)} above threshold {threshold}"
        )

        return filtered[:top_k]

    def rerank_with_diversity(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
        diversity_weight: float = 0.3,
    ) -> List[Dict]:
        if not documents:
            return []

        reranked = self.rerank(query, documents, top_k=len(documents))

        selected = []
        seen_pages = set()

        for doc in reranked:
            if len(selected) >= top_k:
                break

            page_id = doc.get("page_id", "")

            diversity_bonus = 0.0
            if page_id and page_id not in seen_pages:
                diversity_bonus = diversity_weight
                seen_pages.add(page_id)

            doc["final_score"] = (
                doc["rerank_score"] * (1 - diversity_weight) + diversity_bonus
            )
            selected.append(doc)

        selected.sort(key=lambda x: x["final_score"], reverse=True)

        logger.info(
            f"[Reranker] Diversity rerank: {len(seen_pages)} unique sources in top {len(selected)}"
        )

        return selected[:top_k]

    def batch_rerank(
        self, queries: List[str], documents_list: List[List[Dict]], top_k: int = 5
    ) -> List[List[Dict]]:
        logger.info(f"[Reranker] Batch reranking for {len(queries)} queries")

        results = []
        for query, documents in zip(queries, documents_list):
            reranked = self.rerank(query, documents, top_k)
            results.append(reranked)

        return results
