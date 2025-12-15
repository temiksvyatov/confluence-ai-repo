import logging
from typing import Dict, List

import tiktoken

logger = logging.getLogger(__name__)


class ChunkingService:
    def __init__(self, model_name: str = "cl100k_base"):
        """
        Инициализация сервиса чанкинга с токенизацией.

        Args:
            model_name: Название модели токенизации (cl100k_base для GPT-4/3.5)
        """
        logger.info(f"Initializing chunking service with tokenizer: {model_name}")
        try:
            self.tokenizer = tiktoken.get_encoding(model_name)
        except Exception as e:
            logger.warning(
                f"Failed to load {model_name}, falling back to cl100k_base: {e}"
            )
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.info("Chunking service initialized successfully")

    def count_tokens(self, text: str) -> int:
        """
        Подсчитать количество токенов в тексте.

        Args:
            text: Текст для подсчета

        Returns:
            Количество токенов
        """
        return len(self.tokenizer.encode(text))

    def chunk_text_with_context(
        self,
        text: str,
        page_title: str,
        page_metadata: Dict[str, str],
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> List[Dict[str, str]]:
        """
        Разбить текст на чанки по токенам с добавлением контекста страницы.

        Args:
            text: Исходный текст страницы
            page_title: Заголовок страницы
            page_metadata: Метаданные (url, page_id, version)
            chunk_size: Размер чанка в токенах
            overlap: Перекрытие между чанками в токенах

        Returns:
            Список словарей с чанками и метаданными
        """
        logger.info(f"Chunking text from page '{page_title}'...")
        logger.info(
            f"Text length: {len(text)} characters, ~{self.count_tokens(text)} tokens"
        )

        # Создаем контекстный префикс для каждого чанка
        context_prefix = f"Документ: {page_title}\n\n"
        context_prefix_tokens = self.count_tokens(context_prefix)

        # Уменьшаем размер чанка на размер префикса
        effective_chunk_size = chunk_size - context_prefix_tokens

        if effective_chunk_size <= 0:
            logger.error(
                f"Chunk size {chunk_size} too small for context prefix ({context_prefix_tokens} tokens)"
            )
            effective_chunk_size = chunk_size // 2

        # Токенизируем весь текст
        tokens = self.tokenizer.encode(text)

        if len(tokens) <= effective_chunk_size:
            logger.debug("Text is shorter than chunk size. Creating single chunk.")
            full_text = context_prefix + text
            return [
                {
                    "text": full_text,
                    "chunk_index": str(0),
                    "token_count": str(self.count_tokens(full_text)),
                    "has_context": str(True),
                }
            ]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = start + effective_chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Добавляем контекст страницы
            full_chunk_text = context_prefix + chunk_text

            chunks.append(
                {
                    "text": full_chunk_text,
                    "chunk_index": str(chunk_index),
                    "token_count": str(len(chunk_tokens) + context_prefix_tokens),
                    "has_context": str(True),
                }
            )

            chunk_index += 1
            start += effective_chunk_size - overlap

            if end >= len(tokens):
                break

        logger.info(f"Successfully created {len(chunks)} chunks with context")
        logger.info(
            f"Average chunk size: {sum(int(c['token_count']) for c in chunks) / len(chunks):.0f} tokens"
        )

        return chunks

    def chunk_text_semantic(
        self,
        text: str,
        page_title: str,
        page_metadata: Dict[str, str],
        chunk_size: int = 512,
        overlap: int = 50,
        min_chunk_size: int = 100,
    ) -> List[Dict[str, str]]:
        """
        Разбить текст на чанки с учетом смысловых границ (параграфы, предложения).

        Args:
            text: Исходный текст страницы
            page_title: Заголовок страницы
            page_metadata: Метаданные
            chunk_size: Максимальный размер чанка в токенах
            overlap: Перекрытие между чанками в токенах
            min_chunk_size: Минимальный размер чанка в токенах

        Returns:
            Список словарей с чанками и метаданными
        """
        logger.info(f"Semantic chunking for page '{page_title}'...")

        context_prefix = f"Документ: {page_title}\n\n"
        context_prefix_tokens = self.count_tokens(context_prefix)
        effective_chunk_size = chunk_size - context_prefix_tokens

        # Разбиваем на параграфы
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for paragraph in paragraphs:
            para_tokens = self.count_tokens(paragraph)

            # Если параграф сам по себе больше chunk_size, разбиваем его
            if para_tokens > effective_chunk_size:
                # Сохраняем текущий чанк если есть
                if current_chunk:
                    chunk_text = context_prefix + "\n\n".join(current_chunk)
                    chunks.append(
                        {
                            "text": chunk_text,
                            "chunk_index": chunk_index,
                            "token_count": self.count_tokens(chunk_text),
                            "has_context": True,
                        }
                    )
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0

                # Разбиваем большой параграф на предложения
                sentences = [s.strip() + "." for s in paragraph.split(".") if s.strip()]
                for sentence in sentences:
                    sent_tokens = self.count_tokens(sentence)

                    if (
                        current_tokens + sent_tokens > effective_chunk_size
                        and current_chunk
                    ):
                        chunk_text = context_prefix + "\n\n".join(current_chunk)
                        chunks.append(
                            {
                                "text": chunk_text,
                                "chunk_index": chunk_index,
                                "token_count": self.count_tokens(chunk_text),
                                "has_context": True,
                            }
                        )
                        chunk_index += 1

                        # Добавляем overlap
                        if len(current_chunk) > 1:
                            current_chunk = [current_chunk[-1]]
                            current_tokens = self.count_tokens(current_chunk[0])
                        else:
                            current_chunk = []
                            current_tokens = 0

                    current_chunk.append(sentence)
                    current_tokens += sent_tokens

            # Если добавление параграфа превысит размер, сохраняем текущий чанк
            elif current_tokens + para_tokens > effective_chunk_size and current_chunk:
                chunk_text = context_prefix + "\n\n".join(current_chunk)
                chunks.append(
                    {
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "token_count": self.count_tokens(chunk_text),
                        "has_context": True,
                    }
                )
                chunk_index += 1

                # Добавляем overlap
                if len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1]]
                    current_tokens = self.count_tokens(current_chunk[0])
                else:
                    current_chunk = []
                    current_tokens = 0

                current_chunk.append(paragraph)
                current_tokens += para_tokens

            else:
                current_chunk.append(paragraph)
                current_tokens += para_tokens

        # Добавляем последний чанк
        if current_chunk and current_tokens >= min_chunk_size:
            chunk_text = context_prefix + "\n\n".join(current_chunk)
            chunks.append(
                {
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "token_count": self.count_tokens(chunk_text),
                    "has_context": True,
                }
            )

        if not chunks:
            # Если ничего не получилось, возвращаем весь текст
            full_text = context_prefix + text
            chunks = [
                {
                    "text": full_text,
                    "chunk_index": 0,
                    "token_count": self.count_tokens(full_text),
                    "has_context": True,
                }
            ]

        logger.info(f"Successfully created {len(chunks)} semantic chunks with context")
        logger.info(
            f"Average chunk size: {sum(c['token_count'] for c in chunks) / len(chunks):.0f} tokens"
        )

        return chunks
