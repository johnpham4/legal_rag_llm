import requests
import re
import time
from typing import Dict, Optional, List
from bs4 import BeautifulSoup
from loguru import logger

from .base import BaseCrawler
from llm_engineering.domain.documents import Document
from llm_engineering.domain.types import LegalField


class LegalDocumentCrawler(BaseCrawler):

    model = Document

    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en;q=0.8',
        })

    def extract(self, link: str, **kwargs) -> None:
        """Extract legal document using HTML parsing"""

        # Check existing
        existing = self.model.find(link=link)
        if existing:
            logger.info(f"Document exists: {link}")
            return

        logger.info(f"Crawling: {link}")
        time.sleep(2)

        try:
            doc_data = self._crawl_html(link)
            if not doc_data or not doc_data.get('content'):
                logger.warning(f"No content: {link}")
                return

            document = self.model(
                content=doc_data.get('content', ''),
                document_number=doc_data.get('document_number', ''),
                document_type=doc_data.get('document_type', ''),
                field=doc_data.get('category', ''),
                link=link,
                platform="thuvienphapluat.vn",
            )
            document.save()
            logger.info(f"Saved: ({len(doc_data['content'])} chars)")

        except Exception as e:
            logger.error(f"Error: {link} - {e}")
            import traceback
            traceback.print_exc()

    def _crawl_html(self, link: str) -> Optional[Dict]:
        """Crawl and parse HTML content - based on Kaggle approach"""

        try:
            # Get category from URL
            parts = link.split('/')
            category = parts[4] if len(parts) > 4 else ''
            category = self._normalize_field(category)

            # Fetch HTML
            response = self.session.get(link, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')

            # Get full text content from the main document div first
            content_div = soup.select_one('#divContentDoc')
            if not content_div:
                content_div = soup.select_one('#ctl00_Content_ThongTinVB_pnlDocContent')

            if not content_div:
                logger.debug(f"No content div found in {link}")
                return None

            # Extract and clean text
            full_text = content_div.get_text(separator='\n', strip=True)
            full_text = self._clean_text(full_text)

            # Extract title from content (first few lines usually contain title)
            lines = full_text.split('\n')
            title = ' '.join(lines[:5]).strip() if lines else ''  # Get first 5 lines as title

            if len(title) > 200:
                title_match = re.search(r'(BỘ LUẬT|LUẬT|Nghị định|Thông tư|Quyết định)\s+[^\.]+', full_text, re.IGNORECASE)
                title = title_match.group(0) if title_match else title[:200]

            doc_number, doc_type = self._extract_law_metadata(full_text)

            return {
                'title': title,
                'document_number': doc_number,
                'document_type': doc_type,
                'category': category,
                'link': link,
                'content': full_text
            }

        except Exception as e:
            logger.error(f"HTML crawl error {link}: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        if not text:
            return ""

        # If it's HTML, extract text
        if '<' in text:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

        # Remove special characters
        text = text.replace('\xa0', ' ').replace('\r', ' ')
        text = re.sub(r'-{3,}', '', text)
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _extract_law_metadata(self, content: str) -> tuple[str, str]:
        """Extract document number and type from content"""
        if not content:
            return "", ""

        # Extract document type - search anywhere in content
        doc_type_match = re.search(r'(BỘ LUẬT|LUẬT|NGHỊ ĐỊNH|THÔNG TƯ|QUYẾT ĐỊNH|CÔNG VĂN|CHỈ THỊ|NGHỊ QUYẾT)', content, re.IGNORECASE)
        doc_type = doc_type_match.group(1) if doc_type_match else ""

        # Extract document number - search for "Số:" pattern
        number_match = re.search(r'[Ss]ố[:\s]+([^\s,;\n]+(?:/[^\s,;\n]+)*)', content)
        doc_number = number_match.group(1).strip() if number_match else ""

        return doc_number, doc_type

    def _normalize_field(self, url_slug: str) -> str:
        # Clean up spaces around hyphens (common in URLs)
        url_slug = re.sub(r'\s*-\s*', '-', url_slug)
        return LegalField.from_url_slug(url_slug)

