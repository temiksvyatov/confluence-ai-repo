import logging
import re
from typing import Dict, List, Optional

from atlassian import Confluence
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ConfluenceService:
    def __init__(self, url: str, username: str, api_token: str):
        logger.info(f"🔌 Инициализация клиента Confluence: {url}")
        self.client = Confluence(
            url=url, username=username, token=api_token, cloud=False
        )
        logger.info("🔌 Клиент Confluence создан.")

    def fetch_page(self, page_id: str) -> Optional[Dict]:
        logger.info(f"📄 Запрос страницы Confluence: {page_id}")
        try:
            page = self.client.get_page_by_id(
                page_id=page_id, expand="body.storage,version"
            )
            logger.info(f"✅ Страница {page_id} успешно получена.")
            return page
        except Exception as e:
            logger.error(f"❌ Ошибка получения страницы {page_id}: {e}", exc_info=True)
            return None

    def get_all_pages_from_space(self, space_key: str, limit: int = 100) -> List[Dict]:
        logger.info(f"📚 Получение всех страниц из space: {space_key}")
        all_pages = []
        start = 0

        try:
            while True:
                response = self.client.get_all_pages_from_space(
                    space=space_key, start=start, limit=limit, expand="version"
                )

                pages = response if isinstance(response, list) else []

                if not pages:
                    break

                all_pages.extend(pages)
                logger.info(f"Получено {len(pages)} страниц (всего: {len(all_pages)})")

                if len(pages) < limit:
                    break

                start += limit

            logger.info(
                f"✅ Всего получено {len(all_pages)} страниц из space {space_key}"
            )
            return all_pages

        except Exception as e:
            logger.error(
                f"❌ Ошибка получения страниц из space {space_key}: {e}", exc_info=True
            )
            return []

    def get_page_info(self, page_id: str) -> Optional[Dict]:
        try:
            page = self.client.get_page_by_id(page_id=page_id, expand="version")
            if page:
                return {
                    "id": page["id"],
                    "title": page.get("title", ""),
                    "version": page.get("version", {}).get("number", 0),
                }
            return None
        except Exception as e:
            logger.error(f"Ошибка получения информации о странице {page_id}: {e}")
            return None

    def clean_html(self, html_content: str) -> str:
        logger.debug("Очистка HTML контента...")
        soup = BeautifulSoup(html_content, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r" +", " ", text)

        logger.debug("HTML контент успешно очищен")
        return text.strip()

    def extract_page_data(self, page: Dict) -> Dict[str, str]:
        logger.info(f"Извлечение данных со страницы {page.get('id')}")
        title = page.get("title", "Без названия")
        html_content = page.get("body", {}).get("storage", {}).get("value", "")
        text = self.clean_html(html_content)

        base_url = page.get("_links", {}).get("base", "")
        web_ui = page.get("_links", {}).get("webui", "")
        url = f"{base_url}{web_ui}" if base_url and web_ui else ""

        logger.info(f"Данные со страницы {page.get('id')} успешно извлечены")
        return {
            "title": title,
            "text": text,
            "url": url,
            "page_id": page["id"],
            "version": page.get("version", {}).get("number", 0),
        }
