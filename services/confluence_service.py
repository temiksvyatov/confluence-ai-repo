from typing import Dict, Optional
import logging
from atlassian import Confluence
from bs4 import BeautifulSoup
import re

# Set up a logger for this module
logger = logging.getLogger(__name__)

class ConfluenceService:
    """Сервис для получения и обработки страниц Confluence."""

    def __init__(self, url: str, username: str, api_token: str):
        """
        Инициализация клиента Confluence.
        Args:
            url: URL Confluence инстанса
            username: Email пользователя
            api_token: API токен
        """
        logger.info(f"🔌 Инициализация клиента Confluence: {url}")
        self.client = Confluence(
            url=url, username=username, token=api_token, cloud=False
        )
        logger.info("🔌 Клиент Confluence создан.")

    def fetch_page(self, page_id: str) -> Optional[Dict]:
        """
        Получить страницу по ID.
        Args:
            page_id: ID страницы Confluence
        Returns:
            Dict с данными страницы или None при ошибке
        """
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

    def clean_html(self, html_content: str) -> str:
        """
        Очистить HTML контент от тегов и форматирования.

        Args:
            html_content: HTML текст из Confluence

        Returns:
            Очищенный текст
        """
        logger.debug("Очистка HTML контента...")
        # Парсим HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Удаляем script и style теги
        for script in soup(["script", "style"]):
            script.decompose()

        # Получаем текст
        text = soup.get_text()

        # Очищаем множественные пробелы и переносы
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r" +", " ", text)

        logger.debug("HTML контент успешно очищен")
        return text.strip()

    def extract_page_data(self, page: Dict) -> Dict[str, str]:
        """
        Извлечь нужные данные из страницы.

        Args:
            page: Объект страницы от Confluence API

        Returns:
            Dict с title, text, url, page_id
        """
        logger.info(f"Извлечение данных со страницы {page.get('id')}")
        title = page.get("title", "Без названия")
        html_content = page.get("body", {}).get("storage", {}).get("value", "")
        text = self.clean_html(html_content)

        # Формируем URL
        base_url = page.get("_links", {}).get("base", "")
        web_ui = page.get("_links", {}).get("webui", "")
        url = f"{base_url}{web_ui}" if base_url and web_ui else ""

        logger.info(f"Данные со страницы {page.get('id')} успешно извлечены")
        return {"title": title, "text": text, "url": url, "page_id": page["id"]}

