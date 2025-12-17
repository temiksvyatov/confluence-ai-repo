import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Callable, Dict

logger = logging.getLogger(__name__)


class IndexingStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    ERROR = "error"


class IndexingService:
    def __init__(
        self,
        confluence_service,
        embedding_service,
        chunking_service,
        qdrant_service,
        chunk_size: int,
        chunk_overlap: int,
        chunking_strategy: str = "semantic",
    ):
        self.confluence_service = confluence_service
        self.embedding_service = embedding_service
        self.chunking_service = chunking_service
        self.qdrant_service = qdrant_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy

        self.status = IndexingStatus.IDLE
        self.should_stop = False
        self.current_space = None
        self.total_pages = 0
        self.processed_pages = 0
        self.indexed_pages = 0
        self.skipped_pages = 0
        self.failed_pages = 0
        self.current_page_title = ""
        self.error_message = ""
        self.start_time = None
        self.end_time = None

        self._indexing_task = None
        self._progress_callbacks = []

    def register_progress_callback(self, callback: Callable):
        self._progress_callbacks.append(callback)

    def _notify_progress(self):
        for callback in self._progress_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"[Indexing] Error in progress callback: {e}")

    def get_progress(self) -> Dict:
        elapsed = None
        if self.start_time:
            end = self.end_time if self.end_time else datetime.now()
            elapsed = (end - self.start_time).total_seconds()

        return {
            "status": self.status.value,
            "space": self.current_space,
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "indexed_pages": self.indexed_pages,
            "skipped_pages": self.skipped_pages,
            "failed_pages": self.failed_pages,
            "current_page": self.current_page_title,
            "error_message": self.error_message,
            "elapsed_time": elapsed,
            "progress_percent": (self.processed_pages / self.total_pages * 100)
            if self.total_pages > 0
            else 0,
        }

    def is_running(self) -> bool:
        return self.status == IndexingStatus.RUNNING

    def stop(self):
        if self.status == IndexingStatus.RUNNING:
            logger.info("[Indexing] Stopping process...")
            self.should_stop = True
            self.status = IndexingStatus.STOPPING

    async def index_space(self, space_key: str) -> Dict:
        if self.is_running():
            return {"success": False, "message": "Индексация уже выполняется"}

        self.status = IndexingStatus.RUNNING
        self.should_stop = False
        self.current_space = space_key
        self.total_pages = 0
        self.processed_pages = 0
        self.indexed_pages = 0
        self.skipped_pages = 0
        self.failed_pages = 0
        self.current_page_title = ""
        self.error_message = ""
        self.start_time = datetime.now()
        self.end_time = None

        try:
            logger.info(f"[Indexing] Starting space indexing: {space_key}")

            pages = await asyncio.to_thread(
                self.confluence_service.get_all_pages_from_space, space_key
            )

            if not pages:
                self.status = IndexingStatus.ERROR
                self.error_message = f"Не найдено страниц в space {space_key}"
                self.end_time = datetime.now()
                self._notify_progress()
                return {"success": False, "message": self.error_message}

            self.total_pages = len(pages)
            logger.info(f"[Indexing] Found {self.total_pages} pages in {space_key}")
            self._notify_progress()

            for page in pages:
                if self.should_stop:
                    logger.info("[Indexing] Stopped by user")
                    self.status = IndexingStatus.STOPPED
                    self.end_time = datetime.now()
                    self._notify_progress()
                    return {
                        "success": True,
                        "message": "Индексация остановлена",
                        "stats": self.get_progress(),
                    }

                page_id = page.get("id")
                page_title = page.get("title", "Без названия")
                page_version = page.get("version", {}).get("number", 0)

                self.current_page_title = page_title
                self._notify_progress()

                try:
                    indexed_version = await asyncio.to_thread(
                        self.qdrant_service.get_page_version, page_id
                    )

                    if indexed_version is not None and indexed_version >= page_version:
                        logger.info(
                            f"[Indexing] Skipping {page_id} (already indexed, v{indexed_version})"
                        )
                        self.skipped_pages += 1
                        self.processed_pages += 1
                        self._notify_progress()
                        continue

                    if indexed_version is not None:
                        logger.info(
                            f"[Indexing] Updating {page_id} (v{indexed_version} -> v{page_version})"
                        )
                        await asyncio.to_thread(
                            self.qdrant_service.delete_by_page_id, page_id
                        )

                    full_page = await asyncio.to_thread(
                        self.confluence_service.fetch_page, page_id
                    )

                    if not full_page:
                        logger.warning(f"[Indexing] Could not fetch {page_id}")
                        self.failed_pages += 1
                        self.processed_pages += 1
                        self._notify_progress()
                        continue

                    page_data = await asyncio.to_thread(
                        self.confluence_service.extract_page_data, full_page
                    )

                    if self.chunking_strategy == "semantic":
                        chunks_data = await asyncio.to_thread(
                            self.chunking_service.chunk_text_semantic,
                            page_data["text"],
                            page_data["title"],
                            page_data,
                            self.chunk_size,
                            self.chunk_overlap,
                        )
                    else:
                        chunks_data = await asyncio.to_thread(
                            self.chunking_service.chunk_text_with_context,
                            page_data["text"],
                            page_data["title"],
                            page_data,
                            self.chunk_size,
                            self.chunk_overlap,
                        )

                    if not chunks_data:
                        logger.warning(f"[Indexing] No chunks for {page_id}")
                        self.failed_pages += 1
                        self.processed_pages += 1
                        self._notify_progress()
                        continue

                    chunk_texts = [chunk["text"] for chunk in chunks_data]

                    embeddings = await asyncio.to_thread(
                        self.embedding_service.embed_batch_passages, chunk_texts
                    )

                    await asyncio.to_thread(
                        self.qdrant_service.upsert_chunks,
                        chunk_texts,
                        embeddings,
                        {
                            "title": page_data["title"],
                            "url": page_data["url"],
                            "page_id": page_data["page_id"],
                            "version": page_data["version"],
                        },
                    )

                    self.indexed_pages += 1
                    logger.info(
                        f"[Indexing] Success: {page_id} ({len(chunks_data)} chunks)"
                    )

                except Exception as e:
                    logger.error(f"[Indexing] Error on {page_id}: {e}", exc_info=True)
                    self.failed_pages += 1

                finally:
                    self.processed_pages += 1
                    self._notify_progress()

            self.status = IndexingStatus.COMPLETED
            self.end_time = datetime.now()
            self._notify_progress()

            logger.info(f"[Indexing] Space indexing completed: {space_key}")
            return {
                "success": True,
                "message": "Индексация завершена",
                "stats": self.get_progress(),
            }

        except Exception as e:
            logger.error(f"[Indexing] Error during space indexing: {e}", exc_info=True)
            self.status = IndexingStatus.ERROR
            self.error_message = str(e)
            self.end_time = datetime.now()
            self._notify_progress()
            return {
                "success": False,
                "message": f"Ошибка индексации: {str(e)}",
                "stats": self.get_progress(),
            }

    def start_indexing_background(self, space_key: str):
        if self._indexing_task and not self._indexing_task.done():
            return {"success": False, "message": "Индексация уже выполняется"}

        self._indexing_task = asyncio.create_task(self.index_space(space_key))
        return {"success": True, "message": "Индексация запущена"}
