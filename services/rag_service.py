from typing import List, Dict
import logging
from openai import OpenAI
import re
import time

# Set up a logger for this module
logger = logging.getLogger(__name__)


class RAGService:
    """Сервис для RAG: поиск контекста и генерация ответа."""

    def __init__(
        self,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        temperature: float = 0.3,
        max_tokens: int = 8000,
    ):
        """
        Инициализация LLM клиента.

        Args:
            llm_base_url: Base URL для LLM API
            llm_api_key: API ключ
            llm_model: Название модели
            temperature: Температура генерации
            max_tokens: Максимум токенов в ответе
        """
        logger.info(f"Initializing LLM client with model: {llm_model}")
        logger.info(
            f"LLM Configuration: base_url={llm_base_url}, temperature={temperature}, max_tokens={max_tokens}"
        )
        self.client = OpenAI(base_url=llm_base_url, api_key=llm_api_key)
        self.model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info("LLM client initialized successfully")

    def build_prompt(self, question: str, context_chunks: List[Dict]) -> str:
        """
        Построить промпт для LLM.

        Args:
            question: Вопрос пользователя
            context_chunks: Найденные чанки с метаданными

        Returns:
            Промпт для LLM
        """
        start_time = time.time()
        logger.info(f"Building prompt for question: '{question}'")
        logger.info(f"Number of context chunks: {len(context_chunks)}")

        # Формируем контекст из чанков
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            chunk_title = chunk.get("title", "Без названия")
            chunk_text = chunk.get("text", "")
            logger.debug(
                f"Chunk {i}: title='{chunk_title}', text_length={len(chunk_text)} chars"
            )
            context_parts.append(f"[Источник {i}: {chunk_title}]\n{chunk_text}\n")

        context = "\n\n".join(context_parts)

        # Формируем промпт
        prompt = f"""Контекст из корпоративной документации Confluence:

{context}

Вопрос: {question}

Ответ на основе предоставленного контекста:"""

        elapsed_time = time.time() - start_time
        logger.info(f"Prompt built in {elapsed_time:.3f} seconds")
        logger.info(f"Total prompt length: {len(prompt)} characters")
        logger.debug(f"Full prompt content:\n{'=' * 80}\n{prompt}\n{'=' * 80}")

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
        logger.info(f"Context chunks count: {len(context_chunks)}")

        system_prompt = (
            "Ты полезный ассистент по корпоративной документации Confluence. "
            "Отвечай только на основе предоставленного контекста. "
            "Если в контексте нет информации для ответа, так и скажи. "
            "Не выдумывай информацию. "
            "Если в ответе есть ссылки на источники, они должны быть в формате: "
            "[Источник](https://example.com). "
            "Не добавляй ссылки, если они не указаны в контексте."
        )

        user_prompt = self.build_prompt(question, context_chunks)

        try:
            logger.info("Preparing LLM request...")
            logger.debug(f"System prompt:\n{system_prompt}")
            logger.debug(f"User prompt length: {len(user_prompt)} chars")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            logger.info(f"Sending request to LLM model: {self.model}")
            logger.info(
                f"Request parameters: temperature={self.temperature}, max_tokens={self.max_tokens}"
            )

            request_start = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            request_elapsed = time.time() - request_start

            answer = response.choices[0].message.content

            # Подробное логирование ответа
            logger.info(f"✓ LLM request completed in {request_elapsed:.3f} seconds")
            logger.info(f"Response length: {len(answer)} characters")
            logger.info(f"Response content:\n{'=' * 80}\n{answer}\n{'=' * 80}")

            # Логирование метаданных ответа если доступны
            if hasattr(response, "usage"):
                logger.info(f"Token usage: {response.usage}")

            logger.info("=" * 80)
            return answer

        except Exception as e:
            logger.error(f"❌ Error generating answer: {str(e)}", exc_info=True)
            logger.error(f"Error type: {type(e).__name__}")
            logger.info("=" * 80)
            return f"Ошибка генерации ответа: {str(e)}"

    def generate_answer_with_history(
        self, question: str, context_chunks: List[Dict], conversation_history: str
    ) -> str:
        """
        Сгенерировать ответ на вопрос на основе контекста и истории переписки.

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
        logger.info(f"Context chunks count: {len(context_chunks)}")
        logger.info(
            f"Conversation history length: {len(conversation_history)} characters"
        )

        # Подготовка контекста
        context_preparation_start = time.time()
        context_list = []
        for i, chunk in enumerate(context_chunks, 1):
            chunk_title = chunk.get("title", "Без названия")
            chunk_text = chunk.get("text", "")
            logger.debug(
                f"Chunk {i}: title='{chunk_title}', length={len(chunk_text)} chars"
            )
            context_list.append(chunk_text)

        context = "\n\n".join(context_list)
        context_prep_elapsed = time.time() - context_preparation_start
        logger.info(
            f"Context preparation completed in {context_prep_elapsed:.3f} seconds"
        )
        logger.info(f"Total context length: {len(context)} characters")

        # Системный промпт с инструкциями
        system_prompt = f"""Ты - ассистент по корпоративной документации Confluence.
Отвечай только на основе предоставленного контекста.
Если информации недостаточно, скажи, что не знаешь ответа.
Не выдумывай информацию.

Если в ответе есть ссылки на источники, они должны быть в формате:
[Источник](https://example.com)

Не добавляй ссылки, если они не указаны в контексте.

Вот история предыдущих сообщений:
{conversation_history}

Используй историю переписки для контекстуального понимания вопроса."""

        user_prompt = f"Контекст:\n{context}\n\nВопрос: {question}"

        try:
            logger.info("Preparing LLM request with history...")
            logger.debug(f"System prompt:\n{system_prompt}")
            logger.debug(f"User prompt length: {len(user_prompt)} chars")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            logger.info(f"Sending request to LLM model: {self.model}")
            logger.info(
                f"Request parameters: temperature={self.temperature}, max_tokens={self.max_tokens}"
            )

            request_start = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            request_elapsed = time.time() - request_start

            answer = response.choices[0].message.content

            # Подробное логирование ответа
            logger.info(f"✓ LLM request completed in {request_elapsed:.3f} seconds")
            logger.info(f"Response length: {len(answer)} characters")
            logger.info(f"Response content:\n{'=' * 80}\n{answer}\n{'=' * 80}")

            # Логирование метаданных ответа если доступны
            if hasattr(response, "usage"):
                logger.info(f"Token usage: {response.usage}")

            logger.info("Answer with history generated successfully")
            logger.info("=" * 80)
            return answer

        except Exception as e:
            logger.error(
                f"❌ Error generating answer with history: {str(e)}", exc_info=True
            )
            logger.error(f"Error type: {type(e).__name__}")
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
        start_time = time.time()
        logger.info("Generating sources text...")
        logger.info(f"Processing {len(context_chunks)} chunks for sources")

        sources = []
        seen_pages = set()

        for i, chunk in enumerate(context_chunks, 1):
            page_id = chunk.get("page_id", "")
            title = chunk.get("title", "Без названия")
            url = chunk.get("url", "")

            logger.debug(f"Chunk {i}: page_id={page_id}, title='{title}', url={url}")

            if page_id and page_id not in seen_pages:
                seen_pages.add(page_id)

                # Проверяем валидность URL
                if url and re.match(r"^https?://", url):
                    sources.append(f"[{title}]({url})")
                    logger.debug(f"Added source with URL: {title}")
                elif title:
                    sources.append(f"• {title}")
                    logger.debug(f"Added source without URL: {title}")

        elapsed_time = time.time() - start_time

        if sources:
            result = "\n\n**Источники:**\n" + "\n".join(sources)
            logger.info(
                f"Generated {len(sources)} unique sources in {elapsed_time:.3f} seconds"
            )
            logger.debug(f"Sources text:\n{result}")
            return result

        logger.info(f"No sources to generate (completed in {elapsed_time:.3f} seconds)")
        return ""
