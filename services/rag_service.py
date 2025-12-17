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
        logger.info(f"Initializing RAG service with model: {llm_model}")
        logger.info(
            f"Configuration: base_url={llm_base_url}, temperature={temperature}, max_tokens={max_tokens}"
        )
        self.client = OpenAI(base_url=llm_base_url, api_key=llm_api_key)
        self.model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info("RAG service initialized successfully")

    def _log_llm_request(self, operation: str, messages: List[Dict], config: Dict):
        logger.info("=" * 80)
        logger.info(f"LLM REQUEST: {operation}")
        logger.info("=" * 80)
        logger.info(f"Model: {config.get('model')}")
        logger.info(f"Temperature: {config.get('temperature')}")
        logger.info(f"Max Tokens: {config.get('max_tokens')}")
        logger.info("-" * 80)
        logger.info("MESSAGES STRUCTURE:")
        for i, msg in enumerate(messages):
            logger.info(f"Message {i + 1}:")
            logger.info(f"  Role: {msg.get('role')}")
            content = msg.get("content", "")
            if len(content) > 500:
                logger.info(f"  Content (first 500 chars): {content[:500]}...")
                logger.info(f"  Content (last 200 chars): ...{content[-200:]}")
                logger.info(f"  Total content length: {len(content)} characters")
            else:
                logger.info(f"  Content: {content}")
        logger.info("=" * 80)

    def _log_llm_response(self, operation: str, response, elapsed: float):
        logger.info("=" * 80)
        logger.info(f"LLM RESPONSE: {operation}")
        logger.info("=" * 80)
        logger.info(f"Response time: {elapsed:.2f}s")

        if hasattr(response, "usage"):
            logger.info("Token usage:")
            logger.info(f"  Prompt tokens: {response.usage.prompt_tokens}")
            logger.info(f"  Completion tokens: {response.usage.completion_tokens}")
            logger.info(f"  Total tokens: {response.usage.total_tokens}")

        content = response.choices[0].message.content
        logger.info(f"Response length: {len(content)} characters")

        if len(content) > 500:
            logger.info(f"Response (first 500 chars): {content[:500]}...")
            logger.info(f"Response (last 200 chars): ...{content[-200:]}")
        else:
            logger.info(f"Response: {content}")

        logger.info("=" * 80)

    def generate_hyde_document(self, query: str) -> str:
        logger.info(f"[HyDE] Starting generation for query: '{query[:100]}...'")

        hyde_prompt = f"""Напиши краткий, информативный ответ на следующий вопрос, как если бы он был взят из документации.
Не упоминай, что это гипотетический ответ. Пиши утвердительно и конкретно.

Вопрос: {query}

Ответ:"""

        messages = [
            {
                "role": "system",
                "content": "Ты - эксперт по технической документации. Пиши кратко и по существу.",
            },
            {"role": "user", "content": hyde_prompt},
        ]

        config = {
            "model": self.model,
            "temperature": 0.3,
            "max_tokens": 500,
        }

        try:
            self._log_llm_request("HyDE Generation", messages, config)

            start_time = time.time()
            response = self.client.chat.completions.create(messages=messages, **config)
            elapsed = time.time() - start_time

            self._log_llm_response("HyDE Generation", response, elapsed)

            hyde_doc = response.choices[0].message.content
            logger.info(f"[HyDE] Successfully generated document in {elapsed:.2f}s")

            return hyde_doc

        except Exception as e:
            logger.error(f"[HyDE] Error: {e}", exc_info=True)
            logger.info("[HyDE] Fallback to original query")
            return query

    def build_prompt_with_sources(
        self, question: str, context_chunks: List[Dict]
    ) -> str:
        logger.info(
            f"[Prompt Builder] Building prompt with {len(context_chunks)} sources"
        )

        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            chunk_title = chunk.get("title", "Без названия")
            chunk_text = chunk.get("text", "")
            chunk_url = chunk.get("url", "")

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

        logger.info(
            f"[Prompt Builder] Prompt built, total length: {len(prompt)} characters"
        )
        return prompt

    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        logger.info("[Answer Generation] Starting")
        logger.info(f"[Answer Generation] Question: '{question}'")
        logger.info(f"[Answer Generation] Context chunks: {len(context_chunks)}")

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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        config = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            self._log_llm_request("Answer Generation", messages, config)

            start_time = time.time()
            response = self.client.chat.completions.create(messages=messages, **config)
            elapsed = time.time() - start_time

            self._log_llm_response("Answer Generation", response, elapsed)

            answer = response.choices[0].message.content
            logger.info("[Answer Generation] Completed successfully")

            return answer

        except Exception as e:
            logger.error(f"[Answer Generation] Error: {e}", exc_info=True)
            return f"Ошибка генерации ответа: {str(e)}"

    def generate_answer_with_history(
        self, question: str, context_chunks: List[Dict], conversation_history: str
    ) -> str:
        logger.info("[Answer with History] Starting")
        logger.info(f"[Answer with History] Question: '{question}'")
        logger.info(f"[Answer with History] Context chunks: {len(context_chunks)}")
        logger.info(
            f"[Answer with History] History length: {len(conversation_history)} characters"
        )

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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        config = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            self._log_llm_request("Answer with History", messages, config)

            start_time = time.time()
            response = self.client.chat.completions.create(messages=messages, **config)
            elapsed = time.time() - start_time

            self._log_llm_response("Answer with History", response, elapsed)

            answer = response.choices[0].message.content
            logger.info("[Answer with History] Completed successfully")

            return answer

        except Exception as e:
            logger.error(f"[Answer with History] Error: {e}", exc_info=True)
            return "Извините, произошла ошибка при генерации ответа."

    def generate_sources_text(self, context_chunks: List[Dict]) -> str:
        logger.info(f"[Sources] Generating list from {len(context_chunks)} chunks")

        seen_pages = {}

        for i, chunk in enumerate(context_chunks, 1):
            page_id = chunk.get("page_id", "")
            title = chunk.get("title", "Без названия")
            url = chunk.get("url", "")

            if page_id and page_id not in seen_pages:
                seen_pages[page_id] = (i, title, url)

        if not seen_pages:
            return ""

        sources_list = []
        for page_id, (idx, title, url) in seen_pages.items():
            if url and re.match(r"^https?://", url):
                sources_list.append(f"[{idx}] [{title}]({url})")
            else:
                sources_list.append(f"[{idx}] {title}")

        result = "\n\n---\n\n**Источники:**\n" + "\n".join(sources_list)

        logger.info(f"[Sources] Generated list with {len(sources_list)} unique pages")
        return result
