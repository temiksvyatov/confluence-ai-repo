import logging
import re
import time
from typing import Dict, List

from openai import OpenAI

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(
        self,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        temperature: float = 0.3,
        max_tokens: int = 8000,
    ):
        """
        Инициализация LLM клиента с поддержкой HyDE.

        Args:
            llm_base_url: Base URL для LLM API
            llm_api_key: API ключ
            llm_model: Название модели
            temperature: Температура генерации
            max_tokens: Максимум токенов в ответе
        """
        logger.info(f"Initializing RAG service with model: {llm_model}")
        logger.info(
            f"Configuration: base_url={llm_base_url}, temperature={temperature}, max_tokens={max_tokens}"
        )
        self.client = OpenAI(base_url=llm_base_url, api_key=llm_api_key)
        self.model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info("RAG service initialized successfully")

    def generate_hyde_document(self, query: str) -> str:
        """
        Генерировать гипотетический документ для запроса (HyDE).

        HyDE (Hypothetical Document Embeddings) - техника, где мы генерируем
        гипотетический ответ на запрос, затем ищем по эмбеддингу этого ответа.
        Это часто дает лучшие результаты, чем поиск по исходному запросу.

        Args:
            query: Исходный запрос пользователя

        Returns:
            Гипотетический документ (ответ)
        """
        logger.info(f"Generating HyDE document for query: '{query[:50]}...'")

        hyde_prompt = f"""Напиши краткий, информативный ответ на следующий вопрос, как если бы он был взят из документации.
Не упоминай, что это гипотетический ответ. Пиши утвердительно и конкретно.

Вопрос: {query}

Ответ:"""

        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты - эксперт по технической документации. Пиши кратко и по существу.",
                    },
                    {"role": "user", "content": hyde_prompt},
                ],
                temperature=0.5,  # Немного выше для креативности
                max_tokens=300,  # Короткий гипотетический документ
            )

            hyde_doc = response.choices[0].message.content
            elapsed = time.time() - start_time

            logger.info(f"HyDE document generated in {elapsed:.2f}s")
            logger.debug(f"HyDE document: {hyde_doc[:100]}...")

            return hyde_doc

        except Exception as e:
            logger.error(f"Error generating HyDE document: {e}")
            # Fallback: возвращаем исходный запрос
            return query

    def build_prompt_with_sources(
        self, question: str, context_chunks: List[Dict]
    ) -> str:
        """
        Построить промпт с пронумерованными источниками.

        Args:
            question: Вопрос пользователя
            context_chunks: Найденные чанки с метаданными

        Returns:
            Промпт для LLM
        """
        logger.info(
            f"Building prompt with {len(context_chunks)} sources for question: '{question[:50]}...'"
        )

        # Формируем контекст с номерами источников
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            chunk_title = chunk.get("title", "Без названия")
            chunk_text = chunk.get("text", "")
            chunk_url = chunk.get("url", "")

            # Форматируем источник с номером
            source_header = f"[{i}] {chunk_title}"
            if chunk_url:
                source_header += f"\nURL: {chunk_url}"

            context_parts.append(f"{source_header}\n{chunk_text}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""На основе предоставленных документов из корпоративной базы знаний Confluence ответь на вопрос пользователя.

ВАЖНЫЕ ПРАВИЛА:
1. Используй ТОЛЬКО информацию из предоставленных документов
2. При упоминании информации ОБЯЗАТЕЛЬНО указывай номер источника в квадратных скобках, например: [1], [2]
3. Если информация есть в нескольких источниках, укажи все: [1, 3]
4. Если в документах нет ответа на вопрос, честно скажи об этом
5. Не придумывай информацию
6. Будь конкретным и структурированным

ДОКУМЕНТЫ:

{context}

---

ВОПРОС: {question}

ОТВЕТ:"""

        logger.info(f"Prompt built, total length: {len(prompt)} characters")
        return prompt

    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """
        Сгенерировать ответ на основе контекста.

        Args:
            question: Вопрос пользователя
            context_chunks: Найденные чанки

        Returns:
            Ответ LLM
        """
        logger.info("=" * 80)
        logger.info("GENERATE_ANSWER CALLED")
        logger.info(f"Question: '{question}'")
        logger.info(f"Context chunks: {len(context_chunks)}")

        system_prompt = """Ты - помощник по корпоративной документации Confluence.

Твоя задача:
- Отвечать точно на основе предоставленных документов
- ВСЕГДА указывать номера источников [1], [2] и т.д. при упоминании информации
- Структурировать ответ для удобства чтения
- Признавать, если информации недостаточно для ответа

Формат ссылок на источники:
- Одиночная ссылка: "Согласно документации [1], ..."
- Множественные ссылки: "Эта информация подтверждается в нескольких источниках [1, 3]"
- В конце предложения: "... настройка выполняется через панель администратора [2]."
"""

        user_prompt = self.build_prompt_with_sources(question, context_chunks)

        try:
            logger.info("Sending request to LLM...")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            elapsed = time.time() - start_time

            answer = response.choices[0].message.content

            logger.info(f"✓ LLM response received in {elapsed:.2f}s")
            logger.info(f"Response length: {len(answer)} characters")

            if hasattr(response, "usage"):
                logger.info(f"Token usage: {response.usage}")

            logger.info("=" * 80)
            return answer

        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}", exc_info=True)
            logger.info("=" * 80)
            return f"Ошибка генерации ответа: {str(e)}"

    def generate_answer_with_history(
        self, question: str, context_chunks: List[Dict], conversation_history: str
    ) -> str:
        """
        Сгенерировать ответ с учетом истории беседы.

        Args:
            question: Вопрос пользователя
            context_chunks: Найденные чанки
            conversation_history: История переписки

        Returns:
            Сгенерированный ответ
        """
        logger.info("=" * 80)
        logger.info("GENERATE_ANSWER_WITH_HISTORY CALLED")
        logger.info(f"Question: '{question}'")
        logger.info(f"Context chunks: {len(context_chunks)}")

        # Формируем контекст с номерами
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            chunk_title = chunk.get("title", "Без названия")
            chunk_text = chunk.get("text", "")
            chunk_url = chunk.get("url", "")

            source_header = f"[{i}] {chunk_title}"
            if chunk_url:
                source_header += f" ({chunk_url})"

            context_parts.append(f"{source_header}\n{chunk_text}")

        context = "\n\n---\n\n".join(context_parts)

        system_prompt = f"""Ты - помощник по корпоративной документации Confluence.

ИСТОРИЯ БЕСЕДЫ:
{conversation_history}

ПРАВИЛА ОТВЕТА:
1. Учитывай контекст предыдущей беседы для понимания вопроса
2. Отвечай на основе ТОЛЬКО предоставленных документов
3. ОБЯЗАТЕЛЬНО указывай номера источников [1], [2] при использовании информации
4. Если информации недостаточно, честно скажи об этом
5. Будь конкретным и структурированным

Формат ссылок:
- "Как упоминалось [1], ..."
- "Согласно нескольким источникам [1, 3], ..."
- "... процесс описан в документации [2]."
"""

        user_prompt = f"""ДОКУМЕНТЫ:

{context}

---

ВОПРОС: {question}

ОТВЕТ:"""

        try:
            logger.info("Sending request to LLM with history...")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            elapsed = time.time() - start_time

            answer = response.choices[0].message.content

            logger.info(f"✓ LLM response with history received in {elapsed:.2f}s")
            logger.info(f"Response length: {len(answer)} characters")

            if hasattr(response, "usage"):
                logger.info(f"Token usage: {response.usage}")

            logger.info("=" * 80)
            return answer

        except Exception as e:
            logger.error(f"❌ Error generating answer with history: {e}", exc_info=True)
            logger.info("=" * 80)
            return "Извините, произошла ошибка при генерации ответа."

    def generate_sources_text(self, context_chunks: List[Dict]) -> str:
        """
        Сформировать текст со списком источников.

        Args:
            context_chunks: Найденные чанки

        Returns:
            Форматированный текст с источниками
        """
        logger.info(f"Generating sources list from {len(context_chunks)} chunks")

        seen_pages = {}  # page_id -> (index, title, url)

        for i, chunk in enumerate(context_chunks, 1):
            page_id = chunk.get("page_id", "")
            title = chunk.get("title", "Без названия")
            url = chunk.get("url", "")

            if page_id and page_id not in seen_pages:
                seen_pages[page_id] = (i, title, url)

        if not seen_pages:
            return ""

        # Формируем список источников
        sources_list = []
        for page_id, (idx, title, url) in seen_pages.items():
            if url and re.match(r"^https?://", url):
                sources_list.append(f"[{idx}] [{title}]({url})")
            else:
                sources_list.append(f"[{idx}] {title}")

        result = "\n\n---\n\n**Источники:**\n" + "\n".join(sources_list)

        logger.info(f"Generated sources list with {len(sources_list)} unique pages")
        return result

    def extract_query_intent(self, query: str) -> Dict[str, any]:
        """
        Извлечь намерение из запроса для улучшения поиска.

        Args:
            query: Запрос пользователя

        Returns:
            Словарь с информацией о намерении
        """
        intent = {
            "is_question": "?" in query,
            "is_how_to": any(word in query.lower() for word in ["как", "how"]),
            "is_what": any(word in query.lower() for word in ["что", "what", "какой"]),
            "is_why": any(word in query.lower() for word in ["почему", "why", "зачем"]),
            "is_comparison": any(
                word in query.lower() for word in ["сравн", "разниц", "vs", "или"]
            ),
        }
        return intent
