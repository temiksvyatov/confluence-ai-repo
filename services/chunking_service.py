import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)


class ChunkingService:
    def __init__(
        self, model_name: str = "cl100k_base", local_tiktoken_path: str = None
    ):
        """
        Инициализация сервиса чанкинга с токенизацией.

        Args:
            model_name: Название модели токенизации (cl100k_base для GPT-4/3.5)
            local_tiktoken_path: Путь к локальному .tiktoken файлу (для закрытых контуров)
        """
        logger.info(f"Initializing chunking service with tokenizer: {model_name}")
        self.tokenizer = None
        self.use_simple_tokenizer = False

        # Пытаемся загрузить tiktoken
        try:
            import tiktoken

            # Если указан локальный путь, загружаем из него
            if local_tiktoken_path:
                if os.path.exists(local_tiktoken_path):
                    logger.info(
                        f"Loading tokenizer from local file: {local_tiktoken_path}"
                    )
                    self.tokenizer = self._load_local_tokenizer(
                        local_tiktoken_path, model_name, tiktoken
                    )
                    logger.info("✅ Tokenizer loaded from local file successfully")
                else:
                    logger.error(
                        f"❌ Local tokenizer file not found: {local_tiktoken_path}"
                    )
                    raise FileNotFoundError(
                        f"Tokenizer file not found: {local_tiktoken_path}"
                    )
            else:
                # Пытаемся загрузить из стандартных мест
                default_paths = [
                    f"/app/tokenizers/{model_name}.tiktoken",
                    f"/app/.cache/tiktoken/{model_name}.tiktoken",
                    f"./tokenizers/{model_name}.tiktoken",
                    f"./{model_name}.tiktoken",
                ]

                loaded = False
                for path in default_paths:
                    if os.path.exists(path):
                        logger.info(f"Found tokenizer at: {path}")
                        try:
                            self.tokenizer = self._load_local_tokenizer(
                                path, model_name, tiktoken
                            )
                            logger.info(f"✅ Loaded tokenizer from: {path}")
                            loaded = True
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load from {path}: {e}")
                            continue

                # Если не нашли локально, пытаемся загрузить через интернет
                if not loaded:
                    logger.info(
                        "No local tokenizer found, attempting online download..."
                    )
                    self.tokenizer = tiktoken.get_encoding(model_name)
                    logger.info("✅ Tokenizer loaded from online successfully")

        except ImportError:
            logger.warning(
                "⚠️  tiktoken not installed, using simple word-based tokenizer"
            )
            self.use_simple_tokenizer = True
        except Exception as e:
            logger.warning(f"⚠️  Failed to load tiktoken: {e}")
            logger.info("Falling back to simple word-based tokenizer")
            self.use_simple_tokenizer = True

        if self.use_simple_tokenizer:
            logger.info(
                "Using simple tokenizer with ~1.4 tokens per word approximation"
            )

    def _load_local_tokenizer(self, filepath: str, model_name: str, tiktoken_module):
        """
        Загрузить tokenizer из локального файла.

        Args:
            filepath: Путь к .tiktoken файлу (бинарный формат!)
            model_name: Имя модели
            tiktoken_module: Импортированный модуль tiktoken

        Returns:
            Encoding объект
        """
        import base64

        from tiktoken.core import Encoding

        logger.info(f"Reading tokenizer file: {filepath}")

        # Читаем файл
        with open(filepath, "rb") as f:
            contents = f.read()

        logger.info(f"File size: {len(contents)} bytes")

        # Парсим файл
        # Формат: каждая строка "base64_token rank"
        mergeable_ranks = {}

        try:
            # Пробуем парсить как текстовый файл (base64 построчно)
            lines = contents.decode("utf-8").strip().split("\n")
            logger.info(f"Parsing text format file with {len(lines)} lines")

            for line in lines:
                if not line.strip():
                    continue

                parts = line.split()
                if len(parts) != 2:
                    continue

                token_b64, rank_str = parts
                try:
                    token_bytes = base64.b64decode(token_b64)
                    rank = int(rank_str)
                    mergeable_ranks[token_bytes] = rank
                except Exception as e:
                    logger.debug(f"Failed to parse line: {line[:50]}... - {e}")
                    continue

            logger.info(
                f"Successfully parsed {len(mergeable_ranks)} tokens from text format"
            )

        except UnicodeDecodeError:
            # Это бинарный файл - ошибка в документации или формате
            logger.error(
                "File appears to be in binary format, but expected text format"
            )
            logger.error("Please ensure the file is in the correct format:")
            logger.error("Each line should be: base64_token rank")
            logger.error("Example: IQ== 0")
            raise ValueError(
                "Invalid tiktoken file format. "
                "File should be text with lines: 'base64_token rank'"
            )

        if not mergeable_ranks:
            raise ValueError(f"Failed to parse any tokens from {filepath}")

        # Специальные токены для cl100k_base (GPT-4, GPT-3.5-turbo)
        if model_name == "cl100k_base":
            special_tokens = {
                "<|endoftext|>": 100257,
                "<|fim_prefix|>": 100258,
                "<|fim_middle|>": 100259,
                "<|fim_suffix|>": 100260,
                "<|endofprompt|>": 100276,
            }
            pat_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        elif model_name == "p50k_base":
            special_tokens = {"<|endoftext|>": 50256}
            pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        elif model_name == "r50k_base":
            special_tokens = {"<|endoftext|>": 50256}
            pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        else:
            special_tokens = {}
            pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Создаем Encoding объект
        encoding = Encoding(
            name=model_name,
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

        logger.info(f"✅ Successfully created Encoding for {model_name}")

        return encoding

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
