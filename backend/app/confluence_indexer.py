import logging
import re
import html
import base64
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .vector_store import get_vectorstore, clear_persistent_store
from .config import settings
import httpx  # ← вместо atlassian

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

async def reindex_confluence():
    logging.info("="*80)
    logging.info("ЗАПУСК ПЕРЕИНДЕКСАЦИИ CONFLUENCE (Bearer Token режим)")
    logging.info("="*80)

    clear_persistent_store()

    base_url = settings.confluence_url.rstrip("/")
    space_key = settings.confluence_space_key
    auth_str = f"{settings.confluence_username}:{settings.confluence_token}"
    basic_auth = base64.b64encode(auth_str.encode()).decode()

    headers = {
        "Authorization": f"Bearer {basic_auth}",
        "Accept": "application/json",
        "User-Agent": "Confluence-RAG-Bot/1.0"
    }

    async with httpx.AsyncClient(headers=headers, timeout=60.0) as client:

        # 2. Загружаем все страницы через CQL
        docs = []
        start = 0
        limit = 100

        while True:
            cql = f'space = "{space_key}" AND type = page'
            url = f"{base_url}/rest/api/search"
            params = {
                "cql": cql,
                "start": start,
                "limit": limit,
                "expand": "body.storage,version"
            }

            try:
                r = await client.get(url, params=params)
                if r.status_code == 429:
                    logging.warning("Rate limit, ждём 10 сек...")
                    await asyncio.sleep(10)
                    continue
                r.raise_for_status()
                data = r.json()
                results = data.get("results", [])

                if not results:
                    break

                for item in results:
                    page = item.get("content", {})
                    if not page:
                        continue

                    title = page.get("title", "Без названия")
                    page_id = page.get("id")
                    version = page.get("version", {}).get("number", 1)
                    body = page.get("body", {}).get("storage", {}).get("value", "")

                    page_url = f"{base_url}/pages/viewpage.action?pageId={page_id}"

                    clean_text = re.sub(r'<[^>]+>', ' ', body)
                    clean_text = html.unescape(clean_text)
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

                    if len(clean_text) < 50:
                        continue

                    docs.append(Document(
                        page_content=clean_text,
                        metadata={
                            "source": page_url,
                            "title": title,
                            "page_id": page_id,
                            "version": version,
                        }
                    ))

                logging.info(f"Загружено {len(docs)} страниц...")
                start += limit

            except Exception as e:
                logging.error(f"Ошибка при загрузке: {e}")
                break

        if not docs:
            logging.error("НИ ОДНА СТРАНИЦА НЕ ЗАГРУЖЕНА")
            return

        # Нарезка и индексация
        chunks = text_splitter.split_documents(docs)
        chunks = [c for c in chunks if len(c.page_content.strip()) >= 20]

        logging.info(f"Готово чанков: {len(chunks)}")

        vectorstore = get_vectorstore()
        for i in range(0, len(chunks), 100):
            vectorstore.add_documents(chunks[i:i+100])
            logging.info(f"  Индексировано {min(i+100, len(chunks))} / {len(chunks)}")

    logging.info("ПЕРЕИНДЕКСАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
