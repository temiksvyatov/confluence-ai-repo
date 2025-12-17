import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)


class ChunkingService:
    def __init__(
        self, model_name: str = "cl100k_base", local_tiktoken_path: str = None
    ):
        logger.info(f"[Chunking] Initializing with tokenizer: {model_name}")
        self.tokenizer = None
        self.use_simple_tokenizer = False

        try:
            import tiktoken

            if local_tiktoken_path:
                if os.path.exists(local_tiktoken_path):
                    logger.info(f"[Chunking] Loading from: {local_tiktoken_path}")
                    self.tokenizer = self._load_local_tokenizer(
                        local_tiktoken_path, model_name, tiktoken
                    )
                    logger.info("[Chunking] Tokenizer loaded from local file")
                else:
                    logger.error(f"[Chunking] File not found: {local_tiktoken_path}")
                    raise FileNotFoundError(
                        f"Tokenizer file not found: {local_tiktoken_path}"
                    )
            else:
                default_paths = [
                    f"/app/tokenizers/{model_name}.tiktoken",
                    f"/app/.cache/tiktoken/{model_name}.tiktoken",
                    f"./tokenizers/{model_name}.tiktoken",
                    f"./{model_name}.tiktoken",
                ]

                loaded = False
                for path in default_paths:
                    if os.path.exists(path):
                        logger.info(f"[Chunking] Found tokenizer at: {path}")
                        try:
                            self.tokenizer = self._load_local_tokenizer(
                                path, model_name, tiktoken
                            )
                            logger.info(f"[Chunking] Loaded from: {path}")
                            loaded = True
                            break
                        except Exception as e:
                            logger.warning(
                                f"[Chunking] Failed to load from {path}: {e}"
                            )
                            continue

                if not loaded:
                    logger.info("[Chunking] Attempting online download")
                    self.tokenizer = tiktoken.get_encoding(model_name)
                    logger.info("[Chunking] Tokenizer loaded online")

        except ImportError:
            logger.warning("[Chunking] tiktoken not installed, using simple tokenizer")
            self.use_simple_tokenizer = True
        except Exception as e:
            logger.warning(f"[Chunking] Failed to load tiktoken: {e}")
            logger.info("[Chunking] Fallback to simple tokenizer")
            self.use_simple_tokenizer = True

        if self.use_simple_tokenizer:
            logger.info("[Chunking] Using simple tokenizer (~1.4 tokens/word)")

    def _load_local_tokenizer(self, filepath: str, model_name: str, tiktoken_module):
        import base64

        from tiktoken.core import Encoding

        logger.info(f"[Chunking] Reading tokenizer file: {filepath}")

        with open(filepath, "rb") as f:
            contents = f.read()

        logger.info(f"[Chunking] File size: {len(contents)} bytes")

        mergeable_ranks = {}

        try:
            lines = contents.decode("utf-8").strip().split("\n")
            logger.info(f"[Chunking] Parsing {len(lines)} lines")

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
                    logger.debug(
                        f"[Chunking] Failed to parse line: {line[:50]}... - {e}"
                    )
                    continue

            logger.info(f"[Chunking] Parsed {len(mergeable_ranks)} tokens")

        except UnicodeDecodeError:
            logger.error("[Chunking] File in binary format, expected text")
            raise ValueError("Invalid tiktoken file format")

        if not mergeable_ranks:
            raise ValueError(f"Failed to parse tokens from {filepath}")

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

        encoding = Encoding(
            name=model_name,
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

        logger.info(f"[Chunking] Created Encoding for {model_name}")
        return encoding

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def chunk_text_with_context(
        self,
        text: str,
        page_title: str,
        page_metadata: Dict[str, str],
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> List[Dict[str, str]]:
        logger.info(f"[Chunking] Processing page '{page_title}'")
        logger.info(
            f"[Chunking] Text: {len(text)} chars, ~{self.count_tokens(text)} tokens"
        )

        context_prefix = f"Документ: {page_title}\n\n"
        context_prefix_tokens = self.count_tokens(context_prefix)

        effective_chunk_size = chunk_size - context_prefix_tokens

        if effective_chunk_size <= 0:
            logger.error(f"[Chunking] Chunk size {chunk_size} too small")
            effective_chunk_size = chunk_size // 2

        tokens = self.tokenizer.encode(text)

        if len(tokens) <= effective_chunk_size:
            logger.debug("[Chunking] Single chunk")
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

        avg_tokens = sum(int(c["token_count"]) for c in chunks) / len(chunks)
        logger.info(
            f"[Chunking] Created {len(chunks)} chunks, avg {avg_tokens:.0f} tokens"
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
        logger.info(f"[Chunking] Semantic chunking for '{page_title}'")

        context_prefix = f"Документ: {page_title}\n\n"
        context_prefix_tokens = self.count_tokens(context_prefix)
        effective_chunk_size = chunk_size - context_prefix_tokens

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for paragraph in paragraphs:
            para_tokens = self.count_tokens(paragraph)

            if para_tokens > effective_chunk_size:
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

                        if len(current_chunk) > 1:
                            current_chunk = [current_chunk[-1]]
                            current_tokens = self.count_tokens(current_chunk[0])
                        else:
                            current_chunk = []
                            current_tokens = 0

                    current_chunk.append(sentence)
                    current_tokens += sent_tokens

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
            full_text = context_prefix + text
            chunks = [
                {
                    "text": full_text,
                    "chunk_index": 0,
                    "token_count": self.count_tokens(full_text),
                    "has_context": True,
                }
            ]

        avg_tokens = sum(c["token_count"] for c in chunks) / len(chunks)
        logger.info(
            f"[Chunking] Created {len(chunks)} semantic chunks, avg {avg_tokens:.0f} tokens"
        )

        return chunks
