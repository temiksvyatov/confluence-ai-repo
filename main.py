import asyncio
import logging
import uuid

from database import ChatDatabase
from fastapi import FastAPI
from nicegui import app, ui

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

fastapi_app = FastAPI(title="Confluence RAG Chat v2 - Enhanced")

settings = get_settings()
logger.info("=" * 80)
logger.info("CONFLUENCE RAG v2 - ENHANCED INITIALIZATION")
logger.info("=" * 80)

logger.info("Initializing services...")

chat_db = ChatDatabase()
logger.info("✅ Chat database initialized")

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


def get_user_id() -> str:
    if "user_id" not in app.storage.user:
        app.storage.user["user_id"] = str(uuid.uuid4())
    return app.storage.user["user_id"]


def get_current_chat_id() -> int:
    if "current_chat_id" not in app.storage.user:
        user_id = get_user_id()
        chat_id = chat_db.create_chat(user_id, "Новый чат")
        app.storage.user["current_chat_id"] = chat_id
    return app.storage.user["current_chat_id"]


def set_current_chat_id(chat_id: int):
    app.storage.user["current_chat_id"] = chat_id


def create_navigation_drawer():
    left_drawer = ui.left_drawer().classes("bg-gradient-to-b from-blue-50 to-blue-100")
    with left_drawer:
        with ui.column().classes("w-full p-6 gap-6"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("chat_bubble", size="32px").classes("text-blue-600")
                ui.label("Confluence RAG").classes("text-xl font-bold text-blue-700")

            ui.separator().classes("bg-blue-200")

            with ui.column().classes("w-full gap-2"):
                with (
                    ui.element("div")
                    .classes(
                        "w-full p-3 rounded-lg hover:bg-blue-200 cursor-pointer transition-all"
                    )
                    .on("click", lambda: ui.navigate.to("/"))
                ):
                    with ui.row().classes("items-center gap-3"):
                        ui.icon("forum", size="24px").classes("text-blue-600")
                        ui.label("Чат").classes("font-medium text-gray-700")

                with (
                    ui.element("div")
                    .classes(
                        "w-full p-3 rounded-lg hover:bg-blue-200 cursor-pointer transition-all"
                    )
                    .on("click", lambda: ui.navigate.to("/indexing"))
                ):
                    with ui.row().classes("items-center gap-3"):
                        ui.icon("cloud_upload", size="24px").classes("text-blue-600")
                        ui.label("Индексация").classes("font-medium text-gray-700")

                with (
                    ui.element("div")
                    .classes(
                        "w-full p-3 rounded-lg hover:bg-blue-200 cursor-pointer transition-all"
                    )
                    .on("click", lambda: ui.navigate.to("/database"))
                ):
                    with ui.row().classes("items-center gap-3"):
                        ui.icon("storage", size="24px").classes("text-blue-600")
                        ui.label("База страниц").classes("font-medium text-gray-700")

    return left_drawer


@ui.page("/")
def chat_page():
    ui.page_title("Confluence RAG Chat v2")
    ui.query("body").style(
        "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh;"
    )

    left_drawer = create_navigation_drawer()
    user_id = get_user_id()
    current_chat_id = get_current_chat_id()

    with ui.header().classes("bg-white shadow-lg"):
        with ui.row().classes("w-full items-center justify-between p-4"):
            with ui.row().classes("items-center gap-3"):
                ui.button(icon="menu", on_click=lambda: left_drawer.toggle()).props(
                    "flat color=blue-7"
                )
                ui.label("Confluence RAG Chat").classes(
                    "text-2xl font-bold text-gray-800"
                )

            with ui.row().classes("items-center gap-2"):
                ui.button(
                    "Индексация", on_click=lambda: ui.navigate.to("/indexing")
                ).props("flat color=blue-7")
                ui.button("База", on_click=lambda: ui.navigate.to("/database")).props(
                    "flat color=blue-7"
                )

    chats_sidebar = ui.left_drawer(value=False).classes("bg-white shadow-xl")

    def update_chat():
        chat_container.clear()
        messages = chat_db.get_chat_messages(current_chat_id, user_id)

        with chat_container:
            if not messages:
                with ui.card().classes(
                    "w-full max-w-4xl mx-auto my-4 bg-white shadow-xl rounded-2xl"
                ):
                    with ui.column().classes("p-8 gap-6"):
                        with ui.row().classes("items-center gap-3"):
                            ui.icon("info", size="32px").classes("text-blue-600")
                            ui.label("Добро пожаловать в Confluence RAG Chat!").classes(
                                "text-2xl font-bold text-gray-800"
                            )

                        ui.markdown("""
**Новые возможности v2:**
- 🔮 HyDE - улучшенный поиск
- 🎯 Reranking - точная переранжировка
- 📊 Чанкинг по токенам
- 🔗 Улучшенные промпты
- 🚀 BGE-M3 embedding model
- 💾 Сохранение истории чатов
- 📑 Множественные чаты

**Примеры вопросов:**
- Как настроить VPN?
- Какая политика отпусков?
- Инструкции по установке ПО
                        """).classes("text-gray-700 leading-relaxed")
            else:
                for msg in messages:
                    is_user = msg["role"] == "user"
                    with ui.card().classes(
                        f"w-full max-w-4xl mx-auto my-3 shadow-lg rounded-2xl {'ml-auto bg-blue-50' if is_user else 'mr-auto bg-white'}"
                    ):
                        with ui.row().classes("w-full items-start gap-4 p-5"):
                            with ui.avatar(
                                color="blue-7" if is_user else "purple-7", size="lg"
                            ):
                                ui.icon(
                                    "person" if is_user else "smart_toy", size="32px"
                                )

                            with ui.column().classes("flex-1"):
                                ui.label("Вы" if is_user else "Ассистент").classes(
                                    "font-bold text-base text-gray-800"
                                )
                                ui.markdown(msg["content"]).classes(
                                    "text-gray-700 mt-2 leading-relaxed"
                                )

    def refresh_chats_sidebar():
        chats_sidebar.clear()
        chats = chat_db.get_user_chats(user_id)

        with chats_sidebar:
            with ui.column().classes("w-full p-4 gap-3"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Чаты").classes("text-xl font-bold text-gray-800")
                    ui.button(
                        icon="close", on_click=lambda: chats_sidebar.toggle()
                    ).props("flat round")

                ui.button("Новый чат", on_click=lambda: create_new_chat()).props(
                    "color=blue-7 icon=add"
                ).classes("w-full")

                ui.separator()

                for chat in chats:
                    is_current = chat["id"] == current_chat_id
                    with (
                        ui.card()
                        .classes(
                            f"w-full p-3 cursor-pointer hover:shadow-lg transition-all {'bg-blue-100' if is_current else 'bg-white'}"
                        )
                        .on("click", lambda c=chat: switch_chat(c["id"]))
                    ):
                        with ui.column().classes("w-full gap-1"):
                            with ui.row().classes(
                                "w-full items-center justify-between"
                            ):
                                title = chat["title"][:30] + (
                                    "..." if len(chat["title"]) > 30 else ""
                                )
                                ui.label(title).classes(
                                    "font-bold text-sm text-gray-800"
                                )
                                ui.button(
                                    icon="delete",
                                    on_click=lambda e, c=chat: delete_chat(e, c["id"]),
                                ).props("flat round size=sm color=red")

                            ui.label(f"Сообщений: {chat['message_count']}").classes(
                                "text-xs text-gray-600"
                            )

    def create_new_chat():
        nonlocal current_chat_id
        chat_id = chat_db.create_chat(user_id, "Новый чат")
        set_current_chat_id(chat_id)
        current_chat_id = chat_id
        chats_sidebar.toggle()
        refresh_chats_sidebar()
        update_chat()

    def switch_chat(chat_id: int):
        nonlocal current_chat_id
        set_current_chat_id(chat_id)
        current_chat_id = chat_id
        chats_sidebar.toggle()
        update_chat()

    def delete_chat(event, chat_id: int):
        nonlocal current_chat_id
        event.stopPropagation()
        chat_db.delete_chat(chat_id, user_id)

        if chat_id == current_chat_id:
            chats = chat_db.get_user_chats(user_id)
            if chats:
                current_chat_id = chats[0]["id"]
                set_current_chat_id(current_chat_id)
            else:
                new_chat_id = chat_db.create_chat(user_id, "Новый чат")
                current_chat_id = new_chat_id
                set_current_chat_id(new_chat_id)

        refresh_chats_sidebar()
        update_chat()

    async def send_message():
        question = question_input.value.strip()

        if not question:
            ui.notify("Введите вопрос", type="warning")
            return

        logger.info("=" * 80)
        logger.info(f"[USER REQUEST] '{question}'")
        logger.info("=" * 80)

        question_input.value = ""
        chat_db.add_message(current_chat_id, user_id, "user", question)

        messages = chat_db.get_chat_messages(current_chat_id, user_id)
        if len(messages) == 1:
            title = question[:50]
            chat_db.update_chat_title(current_chat_id, user_id, title)

        update_chat()

        spinner_container = None

        try:
            with chat_container:
                spinner_container = ui.card().classes(
                    "w-full max-w-4xl mx-auto my-3 bg-white shadow-lg rounded-2xl"
                )
                with spinner_container:
                    with ui.row().classes("w-full items-center gap-4 p-5"):
                        ui.spinner(size="lg", color="purple-7")
                        ui.label("Ассистент думает...").classes(
                            "text-lg font-medium text-gray-700"
                        )

            if settings.use_hyde:
                logger.info("[Pipeline] Generating HyDE document...")
                hyde_doc = await asyncio.to_thread(
                    rag_service.generate_hyde_document, question
                )
                search_query = hyde_doc
            else:
                search_query = question

            logger.info("[Pipeline] Generating query embedding...")
            question_embedding = await asyncio.to_thread(
                embedding_service.embed_query,
                search_query,
            )

            search_limit = (
                settings.top_k_results
                if settings.use_reranker
                else settings.final_top_k
            )
            logger.info(f"[Pipeline] Searching Qdrant (limit={search_limit})...")
            search_results = await asyncio.to_thread(
                qdrant_service.search,
                query_embedding=question_embedding,
                top_k=search_limit,
            )

            if not search_results:
                logger.warning("[Pipeline] No results found")
                answer = "Извините, я не нашел релевантной информации в базе знаний."
            else:
                logger.info(f"[Pipeline] Found {len(search_results)} initial results")

                if settings.use_reranker and reranker_service:
                    logger.info("[Pipeline] Reranking results...")
                    if settings.use_diversity:
                        search_results = await asyncio.to_thread(
                            reranker_service.rerank_with_diversity,
                            question,
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
                    logger.info(f"[Pipeline] Reranked to {len(search_results)} results")

                messages = chat_db.get_chat_messages(current_chat_id, user_id)
                conversation_history = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in messages[:-1]]
                )

                logger.info("[Pipeline] Generating answer...")
                answer = await asyncio.to_thread(
                    rag_service.generate_answer_with_history,
                    question,
                    search_results,
                    conversation_history,
                )

                sources = rag_service.generate_sources_text(search_results)
                if sources:
                    answer += sources

            if spinner_container:
                spinner_container.delete()

            chat_db.add_message(current_chat_id, user_id, "assistant", answer)
            update_chat()

            logger.info("=" * 80)
            logger.info("[Pipeline] COMPLETED")
            logger.info("=" * 80)

        except Exception as e:
            error_msg = f"Ошибка обработки запроса: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if spinner_container:
                spinner_container.delete()
            chat_db.add_message(current_chat_id, user_id, "assistant", error_msg)
            update_chat()
            ui.notify(error_msg, type="negative")

    with ui.column().classes("flex-1 p-6 overflow-hidden"):
        with ui.row().classes("w-full mb-4"):
            ui.button(
                icon="menu_open",
                on_click=lambda: (chats_sidebar.toggle(), refresh_chats_sidebar()),
            ).props("fab color=blue-7")

        chat_container = ui.column().classes("w-full h-full overflow-y-auto")
        update_chat()

    with ui.footer().classes("bg-white shadow-2xl p-6"):
        with ui.row().classes("w-full max-w-4xl mx-auto gap-3 items-center"):
            question_input = (
                ui.input(placeholder="Введите ваш вопрос...")
                .props("outlined rounded clearable")
                .classes("flex-1 text-lg")
                .on("keydown.enter", send_message)
            )

            ui.button(on_click=send_message).props("fab color=blue-7 icon=send size=lg")


@ui.page("/indexing")
def indexing_page():
    ui.page_title("Индексация Confluence")
    ui.query("body").style(
        "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh;"
    )

    left_drawer = create_navigation_drawer()

    with ui.header().classes("bg-white shadow-lg"):
        with ui.row().classes("w-full items-center justify-between p-4"):
            with ui.row().classes("items-center gap-3"):
                ui.button(icon="menu", on_click=lambda: left_drawer.toggle()).props(
                    "flat color=blue-7"
                )
                ui.label("Индексация страниц Confluence").classes(
                    "text-2xl font-bold text-gray-800"
                )

            with ui.row().classes("items-center gap-2"):
                ui.button("Чат", on_click=lambda: ui.navigate.to("/")).props(
                    "flat color=blue-7"
                )
                ui.button("База", on_click=lambda: ui.navigate.to("/database")).props(
                    "flat color=blue-7"
                )

    async def index_page():
        page_id = page_id_input.value.strip()

        if not page_id:
            ui.notify("Введите Page ID", type="warning")
            return

        logger.info(f"[Indexing] Starting for Page ID: {page_id}")
        status_label.text = "Загрузка страницы..."
        spinner.set_visibility(True)
        progress_bar.set_visibility(True)
        progress_bar.value = 0.1

        try:
            logger.info(f"[Indexing] Fetching page {page_id}...")
            progress_bar.value = 0.2
            page = await asyncio.to_thread(confluence_service.fetch_page, page_id)

            if not page:
                logger.warning(f"[Indexing] Page {page_id} not found")
                ui.notify("Страница не найдена", type="negative")
                progress_bar.set_visibility(False)
                spinner.set_visibility(False)
                return

            logger.info("[Indexing] Extracting page data...")
            progress_bar.value = 0.4
            status_label.text = "Обработка текста..."
            page_data = await asyncio.to_thread(
                confluence_service.extract_page_data, page
            )

            logger.info("[Indexing] Splitting into chunks...")
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
                logger.error(f"[Indexing] Failed to create chunks for {page_id}")
                ui.notify("Не удалось создать чанки", type="negative")
                progress_bar.set_visibility(False)
                spinner.set_visibility(False)
                return

            logger.info(f"[Indexing] Created {len(chunks)} chunks")

            chunks_texts = [chunk["text"] for chunk in chunks]
            logger.info(
                f"[Indexing] Generating embeddings for {len(chunks_texts)} chunks..."
            )
            progress_bar.value = 0.8
            status_label.text = "Генерация эмбеддингов..."
            embeddings = await asyncio.to_thread(
                embedding_service.embed_batch_passages, chunks_texts
            )

            logger.info("[Indexing] Upserting to Qdrant...")
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

            success_msg = f"Успешно проиндексировано {count} чанков"
            logger.info(f"[Indexing] {success_msg} for page '{page_data['title']}'")
            status_label.text = success_msg
            ui.notify(success_msg, type="positive")

            with ui.dialog() as details_dialog, ui.card().classes("w-full max-w-md"):
                ui.label("Детали индексации").classes("text-xl font-bold mb-4")
                with ui.column().classes("w-full gap-2"):
                    ui.label(f"Заголовок: {page_data['title']}").classes("text-sm")
                    ui.label(f"ID страницы: {page_data['page_id']}").classes("text-sm")
                    ui.label(f"Создано чанков: {len(chunks)}").classes("text-sm")
                    ui.label(f"Сохранено в базу: {count}").classes("text-sm")
                ui.button("Закрыть", on_click=details_dialog.close).props(
                    "flat color=blue-7"
                )

            details_dialog.open()

        except Exception as e:
            error_msg = f"Ошибка индексации: {str(e)}"
            logger.error(f"[Indexing] {error_msg}", exc_info=True)
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

        current_page = progress["current_page"]
        if len(current_page) > 50:
            current_page = current_page[:47] + "..."
        space_current_label.text = f"Текущая страница: {current_page}"

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
                "text-lg font-bold"
            )
            ui.label("Это действие необратимо!").classes("text-red-600 mt-2")
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

    with ui.column().classes("flex-1 p-6 overflow-y-auto"):
        with ui.card().classes(
            "w-full max-w-4xl mx-auto p-8 mb-6 bg-white shadow-2xl rounded-2xl"
        ):
            with ui.column().classes("gap-6"):
                ui.label("Индексация одной страницы").classes(
                    "text-2xl font-bold text-gray-800"
                )

                with ui.row().classes("w-full gap-3 items-end"):
                    page_id_input = (
                        ui.input(label="Confluence Page ID", placeholder="123456789")
                        .props("outlined")
                        .classes("flex-1")
                    )

                    ui.button("Индексировать", on_click=index_page).props(
                        "color=blue-7 icon=upload"
                    )

                progress_bar = ui.linear_progress().props("color=blue-7")
                progress_bar.set_visibility(False)

                with ui.row().classes("items-center gap-3"):
                    spinner = ui.spinner(size="md")
                    spinner.set_visibility(False)
                    status_label = ui.label("").classes("text-sm text-gray-600")

        with ui.card().classes(
            "w-full max-w-4xl mx-auto p-8 mb-6 bg-white shadow-2xl rounded-2xl"
        ):
            with ui.column().classes("gap-6"):
                ui.label("Индексация всего Space").classes(
                    "text-2xl font-bold text-gray-800"
                )

                with ui.row().classes("w-full gap-3 items-end"):
                    space_input = (
                        ui.input(label="Space Key", placeholder="MYSPACE")
                        .props("outlined")
                        .classes("flex-1")
                    )

                    start_space_btn = ui.button(
                        "Запустить", on_click=start_space_indexing
                    ).props("color=green icon=play_arrow")
                    stop_space_btn = ui.button(
                        "Остановить", on_click=stop_space_indexing
                    ).props("color=red icon=stop")
                    stop_space_btn.set_enabled(False)

                space_progress_bar = ui.linear_progress().props("color=green")
                space_progress_label = ui.label("Space: N/A").classes(
                    "text-sm font-bold"
                )
                space_stats_label = ui.label(
                    "Обработано: 0/0 | Проиндексировано: 0 | Пропущено: 0 | Ошибок: 0"
                ).classes("text-sm")
                space_current_label = ui.label("Текущая страница: N/A").classes(
                    "text-sm text-gray-600 overflow-hidden text-ellipsis whitespace-nowrap max-w-full"
                )

        with ui.card().classes(
            "w-full max-w-4xl mx-auto p-8 bg-white shadow-2xl rounded-2xl"
        ):
            with ui.column().classes("gap-6"):
                ui.label("Управление базой данных").classes(
                    "text-2xl font-bold text-gray-800"
                )

                ui.button("Очистить всю базу данных", on_click=clear_database).props(
                    "color=red icon=delete"
                )


@ui.page("/database")
def database_page():
    ui.page_title("База проиндексированных страниц")
    ui.query("body").style(
        "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh;"
    )

    left_drawer = create_navigation_drawer()

    with ui.header().classes("bg-white shadow-lg"):
        with ui.row().classes("w-full items-center justify-between p-4"):
            with ui.row().classes("items-center gap-3"):
                ui.button(icon="menu", on_click=lambda: left_drawer.toggle()).props(
                    "flat color=blue-7"
                )
                ui.label("База проиндексированных страниц").classes(
                    "text-2xl font-bold text-gray-800"
                )

            with ui.row().classes("items-center gap-2"):
                ui.button("Чат", on_click=lambda: ui.navigate.to("/")).props(
                    "flat color=blue-7"
                )
                ui.button(
                    "Индексация", on_click=lambda: ui.navigate.to("/indexing")
                ).props("flat color=blue-7")

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
                ui.label("Страницы не найдены").classes(
                    "text-gray-500 text-center p-8 text-lg"
                )
                return

            ui.label(f"Найдено страниц: {len(pages)}").classes(
                "text-lg font-bold mb-4 text-gray-800"
            )

            for page in pages:
                with ui.card().classes(
                    "w-full mb-4 hover:shadow-2xl transition-shadow bg-white rounded-xl"
                ):
                    with ui.row().classes("w-full items-start gap-4 p-5"):
                        ui.icon("description", size="32px").classes("text-blue-600")

                        with ui.column().classes("flex-1"):
                            if page.get("url"):
                                ui.link(
                                    page["title"], page["url"], new_tab=True
                                ).classes(
                                    "text-xl font-bold text-blue-600 hover:text-blue-800"
                                )
                            else:
                                ui.label(page["title"]).classes(
                                    "text-xl font-bold text-gray-800"
                                )

                            ui.label(f"ID: {page['page_id']}").classes(
                                "text-sm text-gray-500 mt-1"
                            )
                            ui.label(f"Версия: {page.get('version', 'N/A')}").classes(
                                "text-sm text-gray-500"
                            )

    async def refresh_pages():
        refresh_btn.set_enabled(False)
        pages = await load_pages()
        display_pages(pages)

        stats = await asyncio.to_thread(qdrant_service.get_collection_stats)
        stats_label.text = f"Всего чанков в базе: {stats['points_count']}"

        refresh_btn.set_enabled(True)
        ui.notify("Данные обновлены", type="positive")

    with ui.column().classes("flex-1 p-6 overflow-y-auto"):
        with ui.card().classes(
            "w-full max-w-5xl mx-auto p-8 bg-white shadow-2xl rounded-2xl"
        ):
            with ui.column().classes("gap-6"):
                ui.label("Поиск по базе страниц").classes(
                    "text-2xl font-bold text-gray-800"
                )

                with ui.row().classes("w-full gap-3"):
                    search_input = (
                        ui.input(placeholder="Введите название страницы или Page ID...")
                        .props("outlined")
                        .classes("flex-1")
                        .on("keydown.enter", search_pages)
                    )

                    ui.button("Поиск", on_click=search_pages).props(
                        "color=blue-7 icon=search"
                    )
                    refresh_btn = ui.button("Обновить", on_click=refresh_pages).props(
                        "color=green icon=refresh"
                    )

                stats_label = ui.label("Загрузка статистики...").classes(
                    "text-sm text-gray-600"
                )

                ui.separator().classes("bg-gray-200")

                results_container = ui.column().classes("w-full gap-3")

    ui.timer(0.1, refresh_pages, once=True)


ui.run_with(
    fastapi_app, mount_path="/", storage_secret="change-this-secret-key-in-production"
)

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server...")
    uvicorn.run("main:fastapi_app", host="0.0.0.0", port=8000, reload=False)
