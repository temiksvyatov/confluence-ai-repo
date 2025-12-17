import logging
import time
from typing import Dict, List, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)


class QdrantService:
    def __init__(self, url: str, collection_name: str, vector_size: int):
        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = None
        self._connect_with_retry()

    def _connect_with_retry(self, max_retries: int = 5, delay: int = 5):
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"[Qdrant] Connecting to {self.url} (attempt {attempt + 1}/{max_retries})"
                )
                self.client = QdrantClient(url=self.url)
                self.client.get_collections()
                logger.info("[Qdrant] Connected successfully")
                return
            except Exception as e:
                logger.error(f"[Qdrant] Connection error: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    logger.warning(f"[Qdrant] Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.critical(f"[Qdrant] Failed after {max_retries} attempts")
                    raise Exception("Не удалось подключиться к Qdrant")

    def init_collection(self):
        logger.info(f"[Qdrant] Initializing collection '{self.collection_name}'")
        try:
            if self.client is None:
                raise ValueError("Client is not initialized")

            collections = self.client.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)

            if not exists:
                logger.info(f"[Qdrant] Creating collection '{self.collection_name}'")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    ),
                )
                logger.info("[Qdrant] Collection created")
            else:
                logger.info("[Qdrant] Collection already exists")
        except Exception as e:
            logger.error(f"[Qdrant] Error initializing collection: {e}", exc_info=True)
            raise

    def get_page_version(self, page_id: str) -> Optional[int]:
        try:
            if self.client is None:
                raise ValueError("Client is not initialized")

            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="page_id", match=MatchValue(value=page_id))
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            if scroll_result[0]:
                return scroll_result[0][0].payload.get("version")
            return None
        except Exception as e:
            logger.error(f"[Qdrant] Error getting version for {page_id}: {e}")
            return None

    def get_all_indexed_pages(self) -> List[Dict[str, any]]:
        try:
            if self.client is None:
                raise ValueError("Client is not initialized")

            pages = {}
            offset = None

            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points, next_offset = scroll_result

                for point in points:
                    page_id = point.payload.get("page_id")
                    if page_id and page_id not in pages:
                        pages[page_id] = {
                            "page_id": page_id,
                            "title": point.payload.get("title", ""),
                            "url": point.payload.get("url", ""),
                            "version": point.payload.get("version", 0),
                        }

                if next_offset is None:
                    break

                offset = next_offset

            logger.info(f"[Qdrant] Retrieved {len(pages)} indexed pages")
            return list(pages.values())

        except Exception as e:
            logger.error(f"[Qdrant] Error getting indexed pages: {e}", exc_info=True)
            return []

    def search_indexed_pages(self, query: str) -> List[Dict[str, any]]:
        try:
            if self.client is None:
                raise ValueError("Client is not initialized")

            pages = {}
            offset = None
            query_lower = query.lower()

            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points, next_offset = scroll_result

                for point in points:
                    page_id = point.payload.get("page_id")
                    title = point.payload.get("title", "")

                    if page_id and page_id not in pages:
                        if (
                            query_lower in title.lower()
                            or query_lower in page_id.lower()
                        ):
                            pages[page_id] = {
                                "page_id": page_id,
                                "title": title,
                                "url": point.payload.get("url", ""),
                                "version": point.payload.get("version", 0),
                            }

                if next_offset is None:
                    break

                offset = next_offset

            logger.info(f"[Qdrant] Search found {len(pages)} pages matching '{query}'")
            return list(pages.values())

        except Exception as e:
            logger.error(f"[Qdrant] Error searching pages: {e}", exc_info=True)
            return []

    def upsert_chunks(
        self, chunks: List[str], embeddings: List[List[float]], metadata: Dict[str, any]
    ) -> int:
        logger.info(
            f"[Qdrant] Upserting {len(chunks)} chunks for page {metadata.get('page_id')}"
        )
        points = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "title": metadata.get("title", ""),
                    "url": metadata.get("url", ""),
                    "page_id": metadata.get("page_id", ""),
                    "version": metadata.get("version", 0),
                    "chunk_index": i,
                },
            )
            points.append(point)

        try:
            if self.client is None:
                raise ValueError("Client is not initialized")

            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"[Qdrant] Successfully upserted {len(points)} points")
            return len(points)
        except Exception as e:
            logger.error(f"[Qdrant] Failed to upsert: {e}", exc_info=True)
            raise

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        page_id_filter: Optional[str] = None,
    ) -> List[Dict]:
        logger.info(f"[Qdrant] Searching for top_k={top_k} results")

        search_filter = None
        if page_id_filter:
            logger.info(f"[Qdrant] Filtering by page_id: {page_id_filter}")
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="page_id", match=MatchValue(value=page_id_filter)
                    )
                ]
            )

        try:
            if self.client is None:
                raise ValueError("Client is not initialized")

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=search_filter,
            ).points

            logger.info(f"[Qdrant] Found {len(results)} matching chunks")
            return [
                {
                    "text": hit.payload.get("text", ""),
                    "title": hit.payload.get("title", ""),
                    "url": hit.payload.get("url", ""),
                    "page_id": hit.payload.get("page_id", ""),
                    "score": hit.score,
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"[Qdrant] Search failed: {e}", exc_info=True)
            return []

    def delete_by_page_id(self, page_id: str) -> bool:
        logger.info(f"[Qdrant] Deleting chunks for page_id: {page_id}")
        try:
            if self.client is None:
                raise ValueError("Client is not initialized")

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="page_id", match=MatchValue(value=page_id))
                    ]
                ),
            )
            logger.info(f"[Qdrant] Deleted chunks for {page_id}")
            return True
        except Exception as e:
            logger.error(f"[Qdrant] Error deleting {page_id}: {e}", exc_info=True)
            return False

    def clear_collection(self) -> bool:
        logger.info(f"[Qdrant] Clearing collection '{self.collection_name}'")
        try:
            if self.client is None:
                raise ValueError("Client is not initialized")

            self.client.delete_collection(collection_name=self.collection_name)
            self.init_collection()
            logger.info("[Qdrant] Collection cleared")
            return True
        except Exception as e:
            logger.error(f"[Qdrant] Error clearing collection: {e}", exc_info=True)
            return False

    def get_collection_stats(self) -> Dict[str, any]:
        try:
            if self.client is None:
                raise ValueError("Client is not initialized")

            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )
            return {
                "points_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count
                if hasattr(collection_info, "indexed_vectors_count")
                else 0,
            }
        except Exception as e:
            logger.error(f"[Qdrant] Error getting stats: {e}")
            return {"points_count": 0, "indexed_vectors_count": 0}
