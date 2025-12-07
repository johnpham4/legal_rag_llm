from zenml import pipeline
from steps.crawl_legal_links import crawl_legal_links


@pipeline
def legal_data_etl(legal_links: list[str]):
    """Pipeline to crawl Vietnamese legal documents"""
    crawl_legal_links(legal_links=legal_links)
