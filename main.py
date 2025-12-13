import asyncio
import logging
from fastapi import FastAPI
from nicegui import ui
from config import get_settings
from services.confluence_service import ConfluenceService
from services.embedding_service import EmbeddingService
from services.qdrant_service import QdrantService
from services.rag_service import RAGService

# --- Logging Configuration ---
# Set up a logger for the application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        # Optionally, you can add a FileHandler to log to a file
        # logging.FileHandler("app.log"),
    ],
)
logger = logging.getLogger(__name__)

# Глобальная переменная для хранения истории сообщений
chat_history = []

# Инициализация FastAPI
app = FastAPI(title="Confluence RAG Chat")

# Настройки
settings = get_settings()
logger.info("Application settings loaded.")

# Инициализация сервисов
logger.info("Initializing services...")

confluence_service = ConfluenceService(
    url=settings.confluence_url,
    username=settings.confluence_username,
    api_token=settings.confluence_api_token,
)

try:
    test_page = confluence_service.fetch_page(
        "836111732"
    )  # Используйте реальный ID тестовой страницы
    if test_page:
        logger.info("✅ Confluence connection successful. Test page found.")
    else:
        logger.warning(
            "⚠️ Confluence connection established, but the test page was not found."
        )
except Exception as e:
    logger.error(f"❌ Error connecting to Confluence: {e}")
    raise  # Прерываем работу, если Confluence недоступен

embedding_service = EmbeddingService(model_name=settings.embedding_model)
logger.info(f"Embedding service initialized with model: {settings.embedding_model}")

qdrant_service = QdrantService(
    url=settings.qdrant_url,
    collection_name=settings.qdrant_collection,
    vector_size=settings.embedding_dimension,
)
logger.info(
    f"Qdrant service initialized for collection: '{settings.qdrant_collection}'"
)

rag_service = RAGService(
    llm_base_url=settings.llm_base_url,
    llm_api_key=settings.llm_api_key,
    llm_model=settings.llm_model,
    temperature=settings.llm_temperature,
    max_tokens=settings.llm_max_tokens,
)
logger.info(f"RAG service initialized with LLM model: {settings.llm_model}")

# Инициализация Qdrant коллекции
try:
    qdrant_service.init_collection()
    logger.info("Qdrant collection initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant collection: {e}")
    raise

logger.info("All services have been successfully initialized.")


@ui.page("/")
def chat_page():
    """Страница чата."""

    # Функция для обновления истории сообщений
    def update_chat_history():
        """Обновить историю сообщений из глобальной переменной."""
        nonlocal messages
        messages = chat_history.copy()
        update_chat()

    # Функция для добавления сообщения в историю
    def add_message(role: str, content: str):
        """Добавить сообщение в историю."""
        chat_history.append({"role": role, "content": content})
        update_chat_history()

    def update_chat():
        """Обновить отображение чата."""
        chat_container.clear()
        with chat_container:
            for msg in messages:
                with ui.card().classes("w-full max-w-3xl mx-auto my-2"):
                    with ui.row().classes("w-full items-start gap-3 p-3"):
                        # Аватар пользователя или ассистента
                        with ui.avatar(
                            color="primary" if msg["role"] == "user" else "secondary"
                        ).classes("shrink"):
                            ui.icon("person" if msg["role"] == "user" else "smart_toy")

                        # Контент сообщения
                        with ui.column().classes("grow"):
                            # Имя отправителя и время
                            with ui.row().classes(
                                "w-full justify-between items-center"
                            ):
                                ui.label(
                                    "Вы" if msg["role"] == "user" else "Ассистент"
                                ).classes("font-bold text-sm")
                                ui.label("только что").classes("text-xs text-gray-500")

                            # Текст сообщения
                            ui.markdown(msg["content"]).classes("text-sm")

    async def send_message():
        """Обработать отправку сообщения."""
        question = question_input.value.strip()

        if not question:
            ui.notify("Введите вопрос", type="warning")
            return

        logger.info(f"Received new question: '{question}'")

        # Очищаем поле ввода
        question_input.value = ""

        # Добавляем вопрос пользователя
        add_message("user", question)

        # Создаем контейнер для спиннера
        spinner_container = None

        try:
            # Показываем спиннер
            with chat_container:
                spinner_container = ui.card().classes("w-full max-w-3xl mx-auto my-2")
                with spinner_container:
                    with ui.row().classes("w-full items-start gap-3 p-3"):
                        with ui.avatar(color="secondary").classes("shrink"):
                            ui.icon("smart_toy")
                        with ui.column().classes("grow"):
                            ui.label("Ассистент").classes("font-bold text-sm")

            # Получаем эмбеддинг вопроса
            logger.info("Generating embedding for the question...")
            question_embedding = await asyncio.to_thread(
                embedding_service.embed, question
            )

            # Ищем в Qdrant
            logger.info(
                f"Searching in Qdrant for top_k={settings.top_k_results} results..."
            )
            search_results = await asyncio.to_thread(
                qdrant_service.search,
                query_embedding=question_embedding,
                top_k=settings.top_k_results,
            )

            if not search_results:
                logger.warning("No relevant information found in the knowledge base.")
                answer = "Извините, я не нашел релевантной информации в базе знаний."
            else:
                logger.info(
                    f"Found {len(search_results)} relevant chunks. Generating answer..."
                )

                # Формируем контекст из истории переписки
                conversation_history = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in chat_history]
                )

                # Генерируем ответ с учетом истории переписки
                answer = await asyncio.to_thread(
                    rag_service.generate_answer_with_history,
                    question,
                    search_results,
                    conversation_history,
                )

                # Добавляем источники
                sources = rag_service.generate_sources_text(search_results)
                if sources:
                    answer += "\n\n" + sources

            # Удаляем спиннер и добавляем ответ
            if spinner_container:
                spinner_container.delete()
            add_message("assistant", answer)

        except Exception as e:
            error_msg = f"An error occurred while processing your request: {str(e)}"
            logger.error(
                error_msg, exc_info=True
            )  # exc_info=True will log the full traceback
            if spinner_container:
                spinner_container.delete()
            add_message("assistant", error_msg)
            ui.notify(error_msg, type="negative")

    # UI
    ui.page_title("Confluence RAG Chat")

    # Добавляем стили для всей страницы
    ui.query("body").style("background-color: #f5f7fa;")

    # Создаем боковую панель навигации (элемент верхнего уровня)
    left_drawer = ui.left_drawer().classes("bg-blue-50")
    with left_drawer:
        with ui.column().classes("w-full p-4 gap-4"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("chat").classes("text-2xl text-blue-600")
                ui.label("Confluence RAG").classes("text-lg font-bold text-blue-600")

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

    # Верхняя панель (элемент верхнего уровня)
    with ui.header().classes("bg-blue-600 text-white shadow-md"):
        with ui.row().classes("w-full items-center justify-between p-4"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("menu").on("click", lambda: left_drawer.toggle())
                ui.label("Confluence RAG Chat").classes("text-xl font-bold")

            with ui.row().classes("items-center gap-2"):
                ui.button(
                    "Индексация", on_click=lambda: ui.navigate.to("/indexing")
                ).props("flat color=white")
                ui.button(
                    "Настройки", on_click=lambda: ui.notify("Настройки в разработке")
                ).props("flat color=white")

    # Основной контент
    with ui.column().classes("flex-grow p-4 overflow-auto"):
        chat_container = ui.column().classes("w-full")

        # Инициализация истории сообщений
        messages = chat_history.copy()

        # Обновляем отображение чата
        update_chat()

        # Приветственное сообщение
        if not messages:
            with chat_container:
                with ui.card().classes("w-full max-w-3xl mx-auto my-2 bg-blue-50"):
                    with ui.column().classes("p-6 gap-4"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("info").classes("text-2xl text-blue-600")
                            ui.label("Добро пожаловать в Confluence RAG Chat!").classes(
                                "text-xl font-bold text-blue-600"
                            )

                        ui.markdown(
                            """
                            Задайте вопрос по корпоративной документации, и я постараюсь найти для вас релевантную информацию.

                            **Примеры вопросов:**
                            - Как настроить VPN?
                            - Политика отпусков
                            - Инструкции по установке ПО
                            """
                        ).classes("text-sm")

    # Поле ввода (элемент верхнего уровня)
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
    """Страница индексации."""

    async def index_page():
        """Индексировать страницу Confluence."""
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
            # Получаем страницу
            logger.info(f"Fetching page {page_id} from Confluence...")
            progress_bar.value = 0.2
            page = await asyncio.to_thread(confluence_service.fetch_page, page_id)

            if not page:
                logger.warning(f"Page with ID {page_id} not found in Confluence.")
                ui.notify("Страница не найдена", type="negative")
                progress_bar.set_visibility(False)
                return

            # Извлекаем данные
            logger.info("Extracting page data (title, text, etc.)...")
            progress_bar.value = 0.4
            status_label.text = "Обработка текста..."
            page_data = await asyncio.to_thread(
                confluence_service.extract_page_data, page
            )

            # Чанкинг
            logger.info("Splitting page text into chunks...")
            progress_bar.value = 0.6
            status_label.text = "Разбиение на чанки..."
            chunks = await asyncio.to_thread(
                embedding_service.chunk_text,
                page_data["text"],
                settings.chunk_size,
                settings.chunk_overlap,
            )

            if not chunks:
                logger.error(f"Failed to create chunks from page {page_id}.")
                ui.notify("Не удалось создать чанки", type="negative")
                progress_bar.set_visibility(False)
                return
            logger.info(f"Created {len(chunks)} chunks.")

            # Генерация эмбеддингов
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            progress_bar.value = 0.8
            status_label.text = f"Генерация эмбеддингов ({len(chunks)} чанков)..."
            embeddings = await asyncio.to_thread(embedding_service.embed_batch, chunks)

            # Загрузка в Qdrant
            logger.info(f"Upserting {len(chunks)} chunks and embeddings into Qdrant...")
            progress_bar.value = 0.9
            status_label.text = "Загрузка в базу..."
            count = await asyncio.to_thread(
                qdrant_service.upsert_chunks,
                chunks,
                embeddings,
                {
                    "title": page_data["title"],
                    "url": page_data["url"],
                    "page_id": page_data["page_id"],
                },
            )

            progress_bar.value = 1.0

            success_msg = (
                f"Successfully indexed {count} chunks for page '{page_data['title']}'"
            )
            logger.info(success_msg)
            status_label.text = f"Успешно проиндексировано {count} чанков!"
            ui.notify(success_msg, type="positive")

            # Показываем детали индексации
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
            # Через небольшую задержку скрываем прогресс-бар
            ui.timer(2.0, lambda: progress_bar.set_visibility(False), once=True)

    # UI
    ui.page_title("Индексация Confluence")

    # Добавляем стили для всей страницы
    ui.query("body").style("background-color: #f5f7fa;")

    # Создаем боковую панель навигации (элемент верхнего уровня)
    left_drawer = ui.left_drawer().classes("bg-blue-50")
    with left_drawer:
        with ui.column().classes("w-full p-4 gap-4"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("chat").classes("text-2xl text-blue-600")
                ui.label("Confluence RAG").classes("text-lg font-bold text-blue-600")

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

    # Верхняя панель (элемент верхнего уровня)
    with ui.header().classes("bg-blue-600 text-white shadow-md"):
        with ui.row().classes("w-full items-center justify-between p-4"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("menu").on("click", lambda: left_drawer.toggle())
                ui.label("Индексация страниц Confluence").classes("text-xl font-bold")

            with ui.row().classes("items-center gap-2"):
                ui.button("Чат", on_click=lambda: ui.navigate.to("/")).props(
                    "flat color=white"
                )
                ui.button(
                    "Настройки", on_click=lambda: ui.notify("Настройки в разработке")
                ).props("flat color=white")

    # Контент страницы
    with ui.column().classes("flex-grow p-4"):
        with ui.card().classes("w-full max-w-3xl mx-auto p-6"):
            with ui.column().classes("gap-4"):
                ui.markdown(
                    """
                    ### Индексация страницы Confluence

                    Введите Page ID страницы для индексации в векторную базу данных.

                    Page ID
                                        """
                )

                with ui.row().classes("w-full gap-2"):
                    page_id_input = (
                        ui.input(label="Confluence Page ID", placeholder="123456789")
                        .props("outlined clearable")
                        .classes("flex-grow")
                    )

                    ui.button(on_click=index_page).props(
                        'fab color=blue-6 icon="upload"'
                    )

                # Прогресс-бар
                progress_bar = ui.linear_progress().props("color=blue-6")
                progress_bar.set_visibility(False)

                # Статус и спиннер
                with ui.row().classes("items-center gap-2"):
                    spinner = ui.spinner(size="sm")
                    spinner.set_visibility(False)
                    status_label = ui.label("").classes("text-sm text-gray-600")

                # Дополнительная информация
                with ui.expansion("Дополнительная информация").props('icon="info"'):
                    ui.markdown(
                        """
                        **Что происходит при индексации:**

                        1. Загрузка страницы из Confluence
                        2. Извлечение текстового содержимого
                        3. Разбиение текста на чанки
                        4. Создание векторных представлений (эмбеддингов)
                        5. Сохранение в векторную базу данных

                        После индексации вы сможете задавать вопросы по содержимому страницы в чате.
                        """
                    ).classes("text-sm")


# Монтируем NiceGUI на FastAPI
ui.run_with(app, mount_path="/", storage_secret="change-this-secret-key-in-production")

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

