import re
from urllib.parse import urlparse
from loguru import logger
from .base import BaseCrawler
from .legal import LegalDocumentCrawler

class CrawlerDispatcher:
    def __init__(self):
        self._crawlers = {}

    @classmethod
    def build(cls) -> "CrawlerDispatcher":
        dispatcher = cls()
        return dispatcher

    def register_vn_legal(self) -> "CrawlerDispatcher":
        self.register("https://thuvienphapluat.vn", LegalDocumentCrawler)
        return self

    def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        parsed_domain = urlparse(domain)
        domain = parsed_domain.netloc
        self._crawlers[r"https://(www\.)?{}/*".format(re.escape(domain))] = crawler

    def get_crawler(self, url: str) -> BaseCrawler:
        for pattern, crawler in self._crawlers.items():
            if re.match(pattern, url):
                return crawler()
        else:
            logger.warning(f"No crawler found for {url}. Defaulting to CustomArticleCrawler.")