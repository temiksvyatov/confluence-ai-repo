import logging
import re
import time
from typing import Dict, List

from openai import OpenAI

logger = logging.getLogger(__name__)

file_logger = logging.getLogger("full_llm_logger")
file_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("/data/llm_full.log", mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
file_logger.addHandler(file_handler)
file_logger.propagate = False


class RAGService:
    def __init__(
        self,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        temperature: float = 0.3,
        max_tokens: int = 8000,
    ):
        logger.info(f"[RAG] Initializing with model: {llm_model}")
        logger.info(
            f"[RAG] Config: base_url={llm_base_url}, temp={temperature}, max_tokens={max_tokens}"
        )
        self.client = OpenAI(base_url=llm_base_url, api_key=llm_api_key)
        self.model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info("[RAG] Service initialized")

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

        file_logger.info("=" * 100)
        file_logger.info(f"LLM REQUEST: {operation}")
        file_logger.info("=" * 100)
        file_logger.info(
            f"Model: {config.get('model')}, Temperature: {config.get('temperature')}, Max Tokens: {config.get('max_tokens')}"
        )
        file_logger.info("-" * 100)
        for i, msg in enumerate(messages):
            file_logger.info(f"Message {i + 1} [Role: {msg.get('role')}]:")
            file_logger.info(msg.get("content", ""))
            file_logger.info("-" * 100)

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

        file_logger.info("=" * 100)
        file_logger.info(f"LLM RESPONSE: {operation}")
        file_logger.info("=" * 100)
        file_logger.info(f"Response time: {elapsed:.2f}s")
        if hasattr(response, "usage"):
            file_logger.info(
                f"Token usage: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}, total={response.usage.total_tokens}"
            )
        file_logger.info("-" * 100)
        file_logger.info("FULL RESPONSE:")
        file_logger.info(content)
        file_logger.info("=" * 100)

    def generate_hyde_document(self, query: str) -> str:
        logger.info(f"[HyDE] Starting for query: '{query[:100]}...'")

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
            logger.info(f"[HyDE] Generated in {elapsed:.2f}s")

            return hyde_doc

        except Exception as e:
            logger.error(f"[HyDE] Error: {e}", exc_info=True)
            file_logger.error(f"[HyDE] Error: {e}")
            return query

    def build_prompt_with_sources(
        self, question: str, context_chunks: List[Dict]
    ) -> str:
        logger.info(f"[Prompt] Building with {len(context_chunks)} sources")

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
2. НЕ используй номера источников в своем ответе (например [1], [2])
3. Если в документах нет ответа на вопрос, честно скажи об этом
4. Не придумывай информацию
5. Будь конкретным и структурированным

ДОКУМЕНТЫ:

{context}

---

ВОПРОС: {question}

ОТВЕТ:"""

        logger.info(f"[Prompt] Built, length: {len(prompt)} chars")
        return prompt

    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        logger.info("[Answer] Starting generation")
        logger.info(f"[Answer] Question: '{question}'")
        logger.info(f"[Answer] Context chunks: {len(context_chunks)}")

        system_prompt = """Ты - помощник по корпоративной документации Confluence.

Твоя задача:
- Отвечать точно на основе предоставленных документов
- НЕ указывай номера источников в тексте ответа
- Структурировать ответ для удобства чтения
- Признавать, если информации недостаточно для ответа
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
            answer = re.sub(r"\[\d+\]", "", answer)
            logger.info("[Answer] Generation completed")

            return answer

        except Exception as e:
            logger.error(f"[Answer] Error: {e}", exc_info=True)
            file_logger.error(f"[Answer] Error: {e}")
            return f"Ошибка генерации ответа: {str(e)}"

    def generate_answer_with_history(
        self, question: str, context_chunks: List[Dict], conversation_history: str
    ) -> str:
        logger.info("[Answer+History] Starting")
        logger.info(f"[Answer+History] Question: '{question}'")
        logger.info(f"[Answer+History] Context chunks: {len(context_chunks)}")
        logger.info(
            f"[Answer+History] History length: {len(conversation_history)} chars"
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
3. НЕ указывай номера источников в тексте ответа
4. Если информации недостаточно, честно скажи об этом
5. Будь конкретным и структурированным
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
            answer = re.sub(r"\[\d+\]", "", answer)
            logger.info("[Answer+History] Completed")

            return answer

        except Exception as e:
            logger.error(f"[Answer+History] Error: {e}", exc_info=True)
            file_logger.error(f"[Answer+History] Error: {e}")
            return "Извините, произошла ошибка при генерации ответа."

    def generate_sources_text(self, context_chunks: List[Dict]) -> str:
        logger.info(f"[Sources] Generating from {len(context_chunks)} chunks")

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

        logger.info(f"[Sources] Generated {len(sources_list)} unique pages")
        return result
