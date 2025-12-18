from loguru import logger
from llm_engineering.infrastructure.db.qdrant import connection

def delete_collections():
    """Delete all collections to allow recreation with new config"""

    collections = ["embedded_chunks", "embedded_queries"]

    for collection_name in collections:
        try:
            connection.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Collection {collection_name} not found or already deleted: {e}")

    logger.info("All collections deleted! Run feature_engineering pipeline to recreate with sparse vectors.")

if __name__ == "__main__":
    delete_collections()
