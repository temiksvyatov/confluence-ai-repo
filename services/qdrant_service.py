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
                    f"Attempting to connect to Qdrant at {self.url} ({attempt + 1}/{max_retries})..."
                )
                self.client = QdrantClient(url=self.url)
                self.client.get_collections()
                logger.info("Successfully connected to Qdrant.")
                return
            except Exception as e:
                logger.error(f"Error connecting to Qdrant: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.critical(
                        f"Failed to connect to Qdrant after {max_retries} attempts."
                    )
                    raise Exception(
                        f"Не удалось подключиться к Qdrant после {max_retries} попыток"
                    )

    def init_collection(self):
        logger.info(f"Initializing collection '{self.collection_name}'...")
        try:
            if self.client is None:
                logger.error("Qdrant client is not initialized")
                raise RuntimeError("Qdrant client is not initialized")

            collections = self.client.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)

            if not exists:
                logger.info(
                    f"Collection '{self.collection_name}' does not exist. Creating..."
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    ),
                )
                logger.info(
                    f"Collection '{self.collection_name}' created successfully."
                )
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            logger.error(
                f"Error initializing collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            raise

    def get_page_version(self, page_id: str) -> Optional[int]:
        try:
            if self.client is None:
                logger.error("Qdrant client is not initialized")
                return None

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
            logger.error(f"Error getting page version for {page_id}: {e}")
            return None

    def get_all_indexed_pages(self) -> List[Dict[str, any]]:
        try:
            if self.client is None:
                logger.error("Qdrant client is not initialized")
                return []

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

            return list(pages.values())

        except Exception as e:
            logger.error(f"Error getting indexed pages: {e}", exc_info=True)
            return []

    def search_indexed_pages(self, query: str) -> List[Dict[str, any]]:
        try:
            pages = {}
            offset = None
            query_lower = query.lower()

            while True:
                if self.client is None:
                    logger.error("Qdrant client is not initialized")
                    return []

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

            return list(pages.values())

        except Exception as e:
            logger.error(f"Error searching indexed pages: {e}", exc_info=True)
            return []

    def upsert_chunks(
        self, chunks: List[str], embeddings: List[List[float]], metadata: Dict[str, any]
    ) -> int:
        logger.info(
            f"Preparing to upsert {len(chunks)} chunks for page_id='{metadata.get('page_id')}'."
        )
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            logger.error(
                "Invalid input: chunks and embeddings must be non-empty and of equal length"
            )
            raise ValueError(
                "Chunks and embeddings must be non-empty and of equal length"
            )

        points = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not isinstance(embedding, list) or not all(
                isinstance(x, (int, float)) for x in embedding
            ):
                logger.error(
                    f"Invalid embedding at index {i}: must be a list of numbers"
                )
                raise ValueError(f"Invalid embedding at index {i}")

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
                logger.error("Qdrant client is not initialized")
                raise RuntimeError("Qdrant client is not initialized")

            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(
                f"Successfully upserted {len(points)} points into collection '{self.collection_name}'."
            )
            return len(points)
        except Exception as e:
            logger.error(
                f"Failed to upsert chunks for page_id='{metadata.get('page_id')}': {e}",
                exc_info=True,
            )
            raise

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        page_id_filter: Optional[str] = None,
    ) -> List[Dict]:
        logger.info(
            f"Searching in collection '{self.collection_name}' for top_k={top_k} results."
        )
        search_filter = None
        if page_id_filter:
            logger.info(f"Applying filter for page_id: {page_id_filter}")
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="page_id", match=MatchValue(value=page_id_filter)
                    )
                ]
            )

        try:
            if self.client is None:
                logger.error("Qdrant client is not initialized")
                return []

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=search_filter,
            )

            logger.info(f"Found {len(results)} matching chunks.")
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
            logger.error(
                f"Search failed in collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            return []

    def delete_by_page_id(self, page_id: str) -> bool:
        logger.info(f"Attempting to delete all chunks for page_id: {page_id}")
        try:
            if self.client is not None:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="page_id", match=MatchValue(value=page_id)
                            )
                        ]
                    ),
                )
                logger.info(f"Successfully deleted chunks for page_id: {page_id}")
                return True
            else:
                logger.error("Qdrant client is not initialized")
                return False
        except Exception as e:
            logger.error(
                f"Error deleting chunks for page_id '{page_id}': {e}", exc_info=True
            )
            return False

    def clear_collection(self) -> bool:
        logger.info(f"Attempting to clear collection '{self.collection_name}'")
        try:
            if self.client is not None:
                self.client.delete_collection(collection_name=self.collection_name)
                self.init_collection()
                logger.info(f"Successfully cleared collection '{self.collection_name}'")
                return True
            else:
                logger.error("Qdrant client is not initialized")
                return False
        except Exception as e:
            logger.error(
                f"Error clearing collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            return False

    def get_collection_stats(self) -> Dict[str, any]:
        try:
            if self.client is None:
                logger.error("Qdrant client is not initialized")
                return {"points_count": 0, "vectors_count": 0}

            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )
            return {
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"points_count": 0, "vectors_count": 0}
