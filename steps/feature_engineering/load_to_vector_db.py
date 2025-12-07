from loguru import logger
from typing_extensions import Annotated
from zenml import step

from llm_engineering.application import utils
from llm_engineering.domain.orm import VectorBaseDocument

@step
def load_to_vector_db(
    documents: Annotated[list, "cleaned documents"]
) -> Annotated[bool, 'successful']:
    logger.info(f"Loading {len(documents)} documents into the vector database.")

    group_documents = VectorBaseDocument.group_by_class(documents)
    for document_cls, documents in group_documents.items():
        logger.info(f"Loading documents into {document_cls.get_collection_name()}")
        for document_batch in utils.misc.batch(documents, size=4):
            try:
                document_cls.bulk_insert(document_batch)
            except Exception:
                logger.error(f"Failed to insert documents into {document_cls.get_collection_name()}")

                return False

    return True
