"""
ZenML step for crawling Vietnamese legal documents
"""
from typing import Annotated
from zenml import step
from tqdm.auto import tqdm
from llm_engineering.application.crawlers.dispatcher import CrawlerDispatcher
from loguru import logger


@step
def crawl_legal_links(
    legal_links: list[str] = []
) -> Annotated[int, "num_crawled"]:
    """Crawl Vietnamese legal documents from provided links"""

    if not legal_links:
        logger.warning("No legal links provided")
        return 0

    dispatcher = CrawlerDispatcher.build().register_vn_legal()

    success_count = 0
    for link in tqdm(legal_links, desc="Crawling links and save to mongodb"):
        try:
            logger.info(f"Processing: {link}")
            crawler = dispatcher.get_crawler(link)
            crawler.extract(link=link)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to crawl {link}: {e}")

    logger.info(f"Successfully crawled {success_count}/{len(legal_links)} documents")
    return success_count
