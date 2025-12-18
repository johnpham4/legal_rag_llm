from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application import utils
from llm_engineering.domain.orm import NoSQLBaseDocument
from llm_engineering.domain.documents import Document

@step
def query_data_warehouse(
    query_limit: int | None = None,
) -> Annotated[list, "raw_documents"]:

    results = fetch_all_data(query_limit=query_limit)

    documents = [doc for query_result in results.values() for doc in query_result]

    logger.info(f"Fetched {len(documents)} documents from data warehouse")

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="raw_documents", metadata=_get_metadata(documents))

    return documents

def fetch_all_data(query_limit: int | None = None) -> dict[str, list[NoSQLBaseDocument]]:
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(_fetch_documents, query_limit): "documents",
        }

        results = {}
        for future in as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                results[query_name] = future.result()
            except Exception:
                logger.exception(f"'{query_name}' request failed.")

                results[query_name] = []

    return results

def _fetch_documents(query_limit: int | None = None) -> list[NoSQLBaseDocument]:
    if query_limit:
        # MongoDB find with limit
        return Document.bulk_find(limit=query_limit)
    return Document.bulk_find()

def _get_metadata(cleaned_documents: list[Document]) -> dict:
    metadata = {
        "num_documents": len(cleaned_documents)
    }
    for document in cleaned_documents:
        _type = document.document_type
        if _type not in metadata:
            metadata[_type] = {}
        metadata[_type]["num_documents"] = metadata[_type].get("num_documents", 0) + 1

    return metadata
