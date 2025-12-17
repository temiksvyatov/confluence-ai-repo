import asyncio
import logging

from fastapi import FastAPI
from nicegui import ui

from config import get_settings
from services.chunking_service import ChunkingService
from services.confluence_service import ConfluenceService
from services.embedding_service import EmbeddingService
from services.indexing_service import IndexingService
from services.qdrant_service import QdrantService
from services.rag_service import RAGService
from services.reranker_service import RerankerService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

chat_history = []

app = FastAPI(title="Confluence RAG Chat v2 - Enhanced")

settings = get_settings()
logger.info("=" * 80)
logger.info("CONFLUENCE RAG v2 - ENHANCED INITIALIZATION")
logger.info("=" * 80)

logger.info("Initializing services...")

confluence_service = ConfluenceService(
    url=settings.confluence_url,
    username=settings.confluence_username,
    api_token=settings.confluence_api_token,
)

try:
    test_page = confluence_service.fetch_page("836111732")
    if test_page:
        logger.info("✅ Confluence connection successful")
except Exception as e:
    logger.error(f"❌ Error connecting to Confluence: {e}")
    raise

embedding_service = EmbeddingService(model_name=settings.embedding_model)
logger.info(f"✅ Embedding service: {settings.embedding_model}")

chunking_service = ChunkingService(
    model_name=settings.tokenizer_model,
    local_tiktoken_path=settings.tiktoken_local_path,
)
logger.info(f"✅ Chunking service: {settings.chunking_strategy} strategy")

if settings.use_reranker:
    reranker_service = RerankerService(model_name=settings.reranker_model)
    logger.info(f"✅ Reranker service: {settings.reranker_model}")
else:
    reranker_service = None
    logger.info("⚠️  Reranker disabled")

qdrant_service = QdrantService(
    url=settings.qdrant_url,
    collection_name=settings.qdrant_collection,
    vector_size=settings.embedding_dimension,
)
logger.info(f"✅ Qdrant service: {settings.qdrant_collection}")

rag_service = RAGService(
    llm_base_url=settings.llm_base_url,
    llm_api_key=settings.llm_api_key,
    llm_model=settings.llm_model,
    temperature=settings.llm_temperature,
    max_tokens=settings.llm_max_tokens,
)
logger.info(f"✅ RAG service: {settings.llm_model}")

indexing_service = IndexingService(
    confluence_service=confluence_service,
    embedding_service=embedding_service,
    chunking_service=chunking_service,
    qdrant_service=qdrant_service,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    chunking_strategy=settings.chunking_strategy,
)
logger.info("✅ Indexing service initialized")

try:
    qdrant_service.init_collection()
    logger.info("✅ Qdrant collection initialized")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant collection: {e}")
    raise

logger.info("=" * 80)
logger.info("ALL SERVICES INITIALIZED")
logger.info("=" * 80)


def create_navigation_drawer():
    left_drawer = ui.left_drawer().classes("bg-blue-50")
    with left_drawer:
        with ui.column().classes("w-full p-4 gap-4"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("chat").classes("text-2xl text-blue-600")
                ui.label("Confluence RAG v2").classes("text-lg font-bold text-blue-600")

            ui.separator()

            with ui.item(on_click=lambda: ui.navigate.to("/")).classes("w-full"):
                with ui.item_section().classes("items-center"):
                    ui.icon("forum")
                with ui.item_section():
                    ui.label("Чат")

            with ui.item(on_click=lambda: ui.navigate.to("/indexing")).classes(
                "w-full"
            ):
                with ui.item_section().classes("items-center"):
                    ui.icon("upload_file")
                with ui.item_section():
                    ui.label("Индексация")

            with ui.item(on_click=lambda: ui.navigate.to("/database")).classes(
                "w-full"
            ):
                with ui.item_section().classes("items-center"):
                    ui.icon("storage")
                with ui.item_section():
                    ui.label("База страниц")

    return left_drawer


@ui.page("/")
def chat_page():
    ui.page_title("Confluence RAG Chat v2")
    ui.query("body").style("background-color: #f5f7fa;")

    left_drawer = create_navigation_drawer()

    with ui.header().classes("bg-blue-600 text-white shadow-md"):
        with ui.row().classes("w-full items-center justify-between p-4"):
            with ui.row().classes("items-center gap-2"):
                ui.button(icon="menu", on_click=lambda: left_drawer.toggle()).props(
                    "flat color=white"
                )
                ui.label("Confluence RAG Chat v2 - Enhanced").classes(
                    "text-xl font-bold"
                )

            with ui.row().classes("items-center gap-2"):
                ui.button(
                    "Индексация", on_click=lambda: ui.navigate.to("/indexing")
                ).props("flat color=white")
                ui.button("База", on_click=lambda: ui.navigate.to("/database")).props(
                    "flat color=white"
                )

    def update_chat_history():
        nonlocal messages
        messages = chat_history.copy()
        update_chat()

    def add_message(role: str, content: str):
        chat_history.append({"role": role, "content": content})
        update_chat_history()

    def update_chat():
        chat_container.clear()
        with chat_container:
            for msg in messages:
                with ui.card().classes("w-full max-w-3xl mx-auto my-2"):
                    with ui.row().classes("w-full items-start gap-3 p-3"):
                        with ui.avatar(
                            color="primary" if msg["role"] == "user" else "secondary"
                        ).classes("shrink"):
                            ui.icon("person" if msg["role"] == "user" else "smart_toy")

                        with ui.column().classes("grow"):
                            with ui.row().classes(
                                "w-full justify-between items-center"
                            ):
                                ui.label(
                                    "Вы" if msg["role"] == "user" else "Ассистент"
                                ).classes("font-bold text-sm")
                                ui.label("только что").classes("text-xs text-gray-500")

                            ui.markdown(msg["content"]).classes("text-sm")

    async def send_message():
        question = question_input.value.strip()

        if not question:
            ui.notify("Введите вопрос", type="warning")
            return

        logger.info("=" * 80)
        logger.info(f"NEW QUESTION: '{question}'")
        logger.info("=" * 80)

        question_input.value = ""
        add_message("user", question)

        spinner_container = None

        try:
            with chat_container:
                spinner_container = ui.card().classes("w-full max-w-3xl mx-auto my-2")
                with spinner_container:
                    with ui.row().classes("w-full items-start gap-3 p-3"):
                        with ui.avatar(color="secondary").classes("shrink"):
                            ui.icon("smart_toy")
                        with ui.column().classes("grow"):
                            ui.label("Ассистент думает...").classes("font-bold text-sm")

            # HyDE: генерируем гипотетический ответ
            if settings.use_hyde:
                logger.info("🔮 Generating HyDE document...")
                hyde_doc = await asyncio.to_thread(
                    rag_service.generate_hyde_document, question
                )
                search_query = hyde_doc
            else:
                search_query = question

            # Генерируем эмбеддинг для ЗАПРОСА (с соответствующим префиксом)
            logger.info("🔍 Generating query embedding...")
            question_embedding = await asyncio.to_thread(
                embedding_service.embed_query,  # Используем метод для запросов
                search_query,
            )

            # Поиск в Qdrant (получаем больше результатов для reranking)
            search_limit = (
                settings.top_k_results
                if settings.use_reranker
                else settings.final_top_k
            )
            logger.info(f"🔍 Searching Qdrant (limit={search_limit})...")
            search_results = await asyncio.to_thread(
                qdrant_service.search,
                query_embedding=question_embedding,
                top_k=search_limit,
            )

            if not search_results:
                logger.warning("❌ No results found")
                answer = "Извините, я не нашел релевантной информации в базе знаний."
            else:
                logger.info(f"✅ Found {len(search_results)} initial results")

                # Reranking
                if settings.use_reranker and reranker_service:
                    logger.info("🎯 Reranking results...")
                    if settings.use_diversity:
                        search_results = await asyncio.to_thread(
                            reranker_service.rerank_with_diversity,
                            question,  # Используем оригинальный вопрос для reranking
                            search_results,
                            top_k=settings.final_top_k,
                            diversity_weight=settings.diversity_weight,
                        )
                    else:
                        search_results = await asyncio.to_thread(
                            reranker_service.rerank,
                            question,
                            search_results,
                            top_k=settings.final_top_k,
                        )
                    logger.info(f"✅ Reranked to {len(search_results)} results")

                # Формируем историю беседы
                conversation_history = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in chat_history]
                )

                # Генерируем ответ с улучшенными промптами
                logger.info("💬 Generating answer...")
                answer = await asyncio.to_thread(
                    rag_service.generate_answer_with_history,
                    question,
                    search_results,
                    conversation_history,
                )

                # Добавляем источники
                sources = rag_service.generate_sources_text(search_results)
                if sources:
                    answer += sources

            if spinner_container:
                spinner_container.delete()
            add_message("assistant", answer)

            logger.info("=" * 80)
            logger.info("QUESTION PROCESSING COMPLETE")
            logger.info("=" * 80)

        except Exception as e:
            error_msg = f"Ошибка обработки запроса: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if spinner_container:
                spinner_container.delete()
            add_message("assistant", error_msg)
            ui.notify(error_msg, type="negative")

    with ui.column().classes("flex-grow p-4 overflow-auto"):
        chat_container = ui.column().classes("w-full")
        messages = chat_history.copy()
        update_chat()

        if not messages:
            with chat_container:
                with ui.card().classes("w-full max-w-3xl mx-auto my-2 bg-blue-50"):
                    with ui.column().classes("p-6 gap-4"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("info").classes("text-2xl text-blue-600")
                            ui.label(
                                "Добро пожаловать в Confluence RAG Chat v2!"
                            ).classes("text-xl font-bold text-blue-600")

                        ui.markdown("""
**Новые возможности v2:**
- 🔮 HyDE - улучшенный поиск через гипотетические документы
- 🎯 Reranking - точная переранжировка результатов
- 📊 Чанкинг по токенам с контекстом страницы
- 🔗 Улучшенные промпты с номерами источников
- 🚀 BGE-M3 embedding model для лучшего качества

**Примеры вопросов:**
- Как настроить VPN?
- Какая политика отпусков?
- Инструкции по установке ПО
                        """).classes("text-sm")

    with ui.footer().classes("bg-white shadow-md p-4"):
        with ui.row().classes("w-full max-w-3xl mx-auto gap-2"):
            question_input = (
                ui.input(placeholder="Введите ваш вопрос...")
                .props("outlined clearable")
                .classes("flex-grow")
                .on("keydown.enter", send_message)
            )

            ui.button(on_click=send_message).props('flat fab color=blue-6 icon="send"')


@ui.page("/indexing")
def indexing_page():
    ui.page_title("Индексация Confluence")
    ui.query("body").style("background-color: #f5f7fa;")

    left_drawer = create_navigation_drawer()

    with ui.header().classes("bg-blue-600 text-white shadow-md"):
        with ui.row().classes("w-full items-center justify-between p-4"):
            with ui.row().classes("items-center gap-2"):
                ui.button(icon="menu", on_click=lambda: left_drawer.toggle()).props(
                    "flat color=white"
                )
                ui.label("Индексация страниц Confluence").classes("text-xl font-bold")

            with ui.row().classes("items-center gap-2"):
                ui.button("Чат", on_click=lambda: ui.navigate.to("/")).props(
                    "flat color=white"
                )
                ui.button("База", on_click=lambda: ui.navigate.to("/database")).props(
                    "flat color=white"
                )

    async def index_page():
        page_id = page_id_input.value.strip()

        if not page_id:
            ui.notify("Введите Page ID", type="warning")
            return

        logger.info(f"Starting indexing for Page ID: {page_id}")
        status_label.text = "Загрузка страницы..."
        spinner.set_visibility(True)
        progress_bar.set_visibility(True)
        progress_bar.value = 0.1

        try:
            logger.info(f"Fetching page {page_id} from Confluence...")
            progress_bar.value = 0.2
            page = await asyncio.to_thread(confluence_service.fetch_page, page_id)

            if not page:
                logger.warning(f"Page with ID {page_id} not found in Confluence.")
                ui.notify("Страница не найдена", type="negative")
                progress_bar.set_visibility(False)
                return

            logger.info("Extracting page data (title, text, etc.)...")
            progress_bar.value = 0.4
            status_label.text = "Обработка текста..."
            page_data = await asyncio.to_thread(
                confluence_service.extract_page_data, page
            )

            logger.info("Splitting page text into chunks...")
            progress_bar.value = 0.6
            status_label.text = "Разбиение на чанки..."

            chunks = await asyncio.to_thread(
                chunking_service.chunk_text_semantic,
                page_data["text"],
                page_data["title"],
                page_data,
                settings.chunk_size,
                settings.chunk_overlap,
            )
            if not chunks:
                logger.error(f"Failed to create chunks from page {page_id}.")
                ui.notify("Не удалось создать чанки", type="negative")
                progress_bar.set_visibility(False)
                return
            logger.info(f"Created {len(chunks)} chunks.")

            chunks_texts = [chunk["text"] for chunk in chunks]
            logger.info(f"Generating embeddings for {len(chunks_texts)} chunks...")
            progress_bar.value = 0.8
            status_label.text = f"Генерация эмбеддингов ({len(chunks_texts)} чанков)..."
            embeddings = await asyncio.to_thread(
                embedding_service.embed_batch_passages, chunks_texts
            )

            logger.info(
                f"Upserting {len(chunks_texts)} chunks and embeddings into Qdrant..."
            )
            progress_bar.value = 0.9
            status_label.text = "Загрузка в базу..."
            count = await asyncio.to_thread(
                qdrant_service.upsert_chunks,
                chunks_texts,
                embeddings,
                {
                    "title": page_data["title"],
                    "url": page_data["url"],
                    "page_id": page_data["page_id"],
                    "version": page_data.get("version", 0),
                },
            )

            progress_bar.value = 1.0

            success_msg = (
                f"Successfully indexed {count} chunks for page '{page_data['title']}'"
            )
            logger.info(success_msg)
            status_label.text = f"Успешно проиндексировано {count} чанков!"
            ui.notify(success_msg, type="positive")

            with ui.dialog() as details_dialog, ui.card().classes("w-full max-w-md"):
                ui.label("Детали индексации").classes("text-h6")
                with ui.column().classes("w-full gap-2"):
                    ui.label(f"Заголовок: {page_data['title']}").classes("text-sm")
                    ui.label(f"ID страницы: {page_data['page_id']}").classes("text-sm")
                    ui.label(f"Создано чанков: {len(chunks)}").classes("text-sm")
                    ui.label(f"Сохранено в базу: {count}").classes("text-sm")
                ui.button("Закрыть", on_click=details_dialog.close).props("flat")

            details_dialog.open()

        except Exception as e:
            error_msg = f"Error during indexing of page {page_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            status_label.text = error_msg
            ui.notify(error_msg, type="negative")
            progress_bar.set_visibility(False)

        finally:
            spinner.set_visibility(False)
            ui.timer(2.0, lambda: progress_bar.set_visibility(False), once=True)

    async def start_space_indexing():
        space_key = space_input.value.strip()

        if not space_key:
            ui.notify("Введите Space Key", type="warning")
            return

        if indexing_service.is_running():
            ui.notify("Индексация уже выполняется", type="warning")
            return

        result = indexing_service.start_indexing_background(space_key)

        if result["success"]:
            ui.notify("Индексация space запущена", type="positive")
            start_space_btn.set_enabled(False)
            stop_space_btn.set_enabled(True)
            space_input.set_enabled(False)
        else:
            ui.notify(result["message"], type="negative")

    def stop_space_indexing():
        indexing_service.stop()
        ui.notify("Остановка индексации...", type="info")

    def update_progress_ui():
        progress = indexing_service.get_progress()

        space_progress_label.text = f"Space: {progress['space'] or 'N/A'}"
        space_stats_label.text = (
            f"Обработано: {progress['processed_pages']}/{progress['total_pages']} | "
            f"Проиндексировано: {progress['indexed_pages']} | "
            f"Пропущено: {progress['skipped_pages']} | "
            f"Ошибок: {progress['failed_pages']}"
        )
        space_current_label.text = f"Текущая страница: {progress['current_page']}"

        if progress["total_pages"] > 0:
            space_progress_bar.value = progress["progress_percent"] / 100

        status = progress["status"]

        if status in ["completed", "stopped", "error"]:
            start_space_btn.set_enabled(True)
            stop_space_btn.set_enabled(False)
            space_input.set_enabled(True)

            if status == "completed":
                ui.notify("Индексация space завершена", type="positive")
            elif status == "stopped":
                ui.notify("Индексация остановлена", type="info")
            elif status == "error":
                ui.notify(f"Ошибка: {progress['error_message']}", type="negative")

    indexing_service.register_progress_callback(update_progress_ui)

    async def clear_database():
        with ui.dialog() as confirm_dialog, ui.card():
            ui.label("Вы уверены, что хотите очистить всю базу данных?").classes(
                "text-h6"
            )
            ui.label("Это действие необратимо!").classes("text-red-600")
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Отмена", on_click=confirm_dialog.close).props("flat")
                ui.button(
                    "Очистить", on_click=lambda: perform_clear(confirm_dialog)
                ).props("color=red")

        confirm_dialog.open()

    async def perform_clear(dialog):
        dialog.close()
        try:
            success = await asyncio.to_thread(qdrant_service.clear_collection)
            if success:
                ui.notify("База данных очищена", type="positive")
            else:
                ui.notify("Ошибка очистки базы данных", type="negative")
        except Exception as e:
            ui.notify(f"Ошибка: {str(e)}", type="negative")

    with ui.column().classes("flex-grow p-4"):
        with ui.card().classes("w-full max-w-3xl mx-auto p-6 mb-4"):
            with ui.column().classes("gap-4"):
                ui.markdown("### Индексация одной страницы")

                with ui.row().classes("w-full gap-2"):
                    page_id_input = (
                        ui.input(label="Confluence Page ID", placeholder="123456789")
                        .props("outlined clearable")
                        .classes("flex-grow")
                    )

                    ui.button(on_click=index_page).props(
                        'fab color=blue-6 icon="upload"'
                    )

                progress_bar = ui.linear_progress().props("color=blue-6")
                progress_bar.set_visibility(False)

                with ui.row().classes("items-center gap-2"):
                    spinner = ui.spinner(size="sm")
                    spinner.set_visibility(False)
                    status_label = ui.label("").classes("text-sm text-gray-600")

        with ui.card().classes("w-full max-w-3xl mx-auto p-6 mb-4"):
            with ui.column().classes("gap-4"):
                ui.markdown("### Индексация всего Space")

                with ui.row().classes("w-full gap-2 items-end"):
                    space_input = (
                        ui.input(label="Space Key", placeholder="MYSPACE")
                        .props("outlined clearable")
                        .classes("flex-grow")
                    )

                    start_space_btn = ui.button(
                        "Запустить", on_click=start_space_indexing
                    ).props("color=green")
                    stop_space_btn = ui.button(
                        "Остановить", on_click=stop_space_indexing
                    ).props("color=red")
                    stop_space_btn.set_enabled(False)

                space_progress_bar = ui.linear_progress().props("color=green")
                space_progress_label = ui.label("Space: N/A").classes(
                    "text-sm font-bold"
                )
                space_stats_label = ui.label(
                    "Обработано: 0/0 | Проиндексировано: 0 | Пропущено: 0 | Ошибок: 0"
                ).classes("text-sm")
                space_current_label = ui.label("Текущая страница: N/A").classes(
                    "text-sm text-gray-600"
                )

        with ui.card().classes("w-full max-w-3xl mx-auto p-6"):
            with ui.column().classes("gap-4"):
                ui.markdown("### Управление базой данных")

                ui.button("Очистить всю базу данных", on_click=clear_database).props(
                    "color=red icon=delete"
                )

                with ui.expansion("Дополнительная информация").props('icon="info"'):
                    ui.markdown("""
**Индексация одной страницы:**
- Введите Page ID для индексации отдельной страницы

**Индексация всего Space:**
- Автоматически индексирует все страницы в указанном Space
- Инкрементальная: пропускает уже проиндексированные страницы
- Может быть остановлена в любой момент
- Не блокирует работу чата

**Очистка базы:**
- Полностью удаляет все проиндексированные данные
- Действие необратимо
                    """).classes("text-sm")


@ui.page("/database")
def database_page():
    ui.page_title("База проиндексированных страниц")
    ui.query("body").style("background-color: #f5f7fa;")

    left_drawer = create_navigation_drawer()

    with ui.header().classes("bg-blue-600 text-white shadow-md"):
        with ui.row().classes("w-full items-center justify-between p-4"):
            with ui.row().classes("items-center gap-2"):
                ui.button(icon="menu", on_click=lambda: left_drawer.toggle()).props(
                    "flat color=white"
                )
                ui.label("База проиндексированных страниц").classes("text-xl font-bold")

            with ui.row().classes("items-center gap-2"):
                ui.button("Чат", on_click=lambda: ui.navigate.to("/")).props(
                    "flat color=white"
                )
                ui.button(
                    "Индексация", on_click=lambda: ui.navigate.to("/indexing")
                ).props("flat color=white")

    async def load_pages():
        try:
            pages = await asyncio.to_thread(qdrant_service.get_all_indexed_pages)
            return pages
        except Exception as e:
            logger.error(f"Error loading pages: {e}")
            ui.notify("Ошибка загрузки страниц", type="negative")
            return []

    async def search_pages():
        query = search_input.value.strip()

        if not query:
            pages = await load_pages()
        else:
            try:
                pages = await asyncio.to_thread(
                    qdrant_service.search_indexed_pages, query
                )
            except Exception as e:
                logger.error(f"Error searching pages: {e}")
                ui.notify("Ошибка поиска", type="negative")
                pages = []

        display_pages(pages)

    def display_pages(pages):
        results_container.clear()

        with results_container:
            if not pages:
                ui.label("Страницы не найдены").classes("text-gray-500 text-center p-4")
                return

            ui.label(f"Найдено страниц: {len(pages)}").classes("text-sm font-bold mb-4")

            for page in pages:
                with ui.card().classes("w-full mb-2 hover:shadow-lg transition-shadow"):
                    with ui.row().classes("w-full items-start gap-3 p-3"):
                        ui.icon("description").classes("text-blue-600 text-2xl")

                        with ui.column().classes("grow"):
                            if page.get("url"):
                                ui.link(
                                    page["title"], page["url"], new_tab=True
                                ).classes("text-lg font-bold text-blue-600")
                            else:
                                ui.label(page["title"]).classes("text-lg font-bold")

                            ui.label(f"ID: {page['page_id']}").classes(
                                "text-xs text-gray-500"
                            )
                            ui.label(f"Версия: {page.get('version', 'N/A')}").classes(
                                "text-xs text-gray-500"
                            )

    async def refresh_pages():
        refresh_btn.set_enabled(False)
        pages = await load_pages()
        display_pages(pages)

        stats = await asyncio.to_thread(qdrant_service.get_collection_stats)
        stats_label.text = f"Всего чанков в базе: {stats['points_count']}"

        refresh_btn.set_enabled(True)
        ui.notify("Данные обновлены", type="positive")

    with ui.column().classes("flex-grow p-4"):
        with ui.card().classes("w-full max-w-4xl mx-auto p-6"):
            with ui.column().classes("gap-4"):
                ui.markdown("### Поиск по базе страниц")

                with ui.row().classes("w-full gap-2"):
                    search_input = (
                        ui.input(placeholder="Введите название страницы или Page ID...")
                        .props("outlined clearable")
                        .classes("flex-grow")
                        .on("keydown.enter", search_pages)
                    )

                    ui.button("Поиск", on_click=search_pages).props(
                        "color=blue-6 icon=search"
                    )
                    refresh_btn = ui.button("Обновить", on_click=refresh_pages).props(
                        "color=green icon=refresh"
                    )

                stats_label = ui.label("Загрузка статистики...").classes(
                    "text-sm text-gray-600"
                )

                ui.separator()

                results_container = ui.column().classes("w-full gap-2")

    ui.timer(0.1, refresh_pages, once=True)


ui.run_with(app, mount_path="/", storage_secret="change-this-secret-key-in-production")

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
