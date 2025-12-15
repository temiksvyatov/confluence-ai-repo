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

chunking_service = ChunkingService(model_name=settings.tokenizer_model)
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
    # Импортируем страницу индексации из старого кода
    # (она работает без изменений, только чанкинг внутри обновлен)
    from main_indexing import create_indexing_page

    create_indexing_page(
        left_drawer=create_navigation_drawer(),
        confluence_service=confluence_service,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        indexing_service=indexing_service,
        settings=settings,
    )


@ui.page("/database")
def database_page():
    # Импортируем страницу базы данных из старого кода
    # (она работает без изменений)
    from main_database import create_database_page

    create_database_page(
        left_drawer=create_navigation_drawer(), qdrant_service=qdrant_service
    )


ui.run_with(app, mount_path="/", storage_secret="change-this-secret-key-in-production")

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
