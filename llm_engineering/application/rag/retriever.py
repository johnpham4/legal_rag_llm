from loguru import logger
import concurrent.futures
from qdrant_client.models import FieldCondition, Filter, MatchValue

from llm_engineering.application.rag.query_expansion import QueryExpansion
from llm_engineering.application.rag.reranking import Reranker
from llm_engineering.application.rag.self_query import SelfQuery
from llm_engineering.application.preprocessing.dispatchers import EmbeddingDispatcher
from llm_engineering.domain.queries import EmbeddedQuery, Query
from llm_engineering.application import utils
from llm_engineering.domain.embedded_chunks import EmbeddedChunk


class ContextRetriever:
    def __init__(self, mock: bool = False) -> None:
        self._query_expander = QueryExpansion(mock=mock)
        self._metadata_extractor = SelfQuery(mock=mock)
        self._reranker = Reranker(mock=mock)
        self._embedding_dispatcher = EmbeddingDispatcher()

    def search(
        self,
        query: str,
        k: int = 3,
        expand_to_n_queries: int = 3,
        use_sparse: bool = True
    ) -> list:
        query_model = Query.from_str(query)
        query_model = self._metadata_extractor.generate(query_model)

        n_generated_queries = self._query_expander.generate(query_model, expand_to_n=expand_to_n_queries)
        logger.info(
            "Successfully generated queries for search.",
            num_queries=len(n_generated_queries),
            use_sparse=use_sparse
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [executor.submit(
                self._search, _query_model, k, use_sparse) for _query_model in n_generated_queries
            ]

            n_k_documents = [task.result() for task in concurrent.futures.as_completed(search_tasks)]
            n_k_documents = utils.misc.flatten(n_k_documents)
            n_k_documents = list(set(n_k_documents))

        logger.info(f"{len(n_k_documents)} documents retrieved successfully")

        if len(n_k_documents) > 0:
            k_documents = self.rerank(query, chunks=n_k_documents, keep_top_k=k)
        else:
            k_documents = []

        return k_documents

    def _search(self, query: Query, k: int = 3, use_sparse: bool = True) -> list[EmbeddedChunk]:
        assert k >= 3, "k should be >= 3"

        # Embed query using EmbeddingDispatcher
        embedded_query: EmbeddedQuery = self._embedding_dispatcher.embed_query(query, use_sparse=use_sparse)

        # Build Qdrant filters from metadata
        query_filter = self._build_filter(query.metadata)

        # Use hybrid search if sparse embedding available
        if embedded_query.sparse_embedding:
            search_results = EmbeddedChunk.hybrid_search(
                query_vector=embedded_query.embedding,
                sparse_query_vector=embedded_query.sparse_embedding,
                limit=k // 3,
                query_filter=query_filter
            )
        else:
            # Fallback to dense-only search
            search_results = EmbeddedChunk.search(
                query_vector=embedded_query.embedding,
                limit=k // 3,
                query_filter=query_filter
            )

        # FALLBACK: If filter returns 0 chunks, retry without filter
        if len(search_results) == 0 and query_filter is not None:
            logger.warning(f"Metadata filter returned 0 chunks, retrying without filter")
            if embedded_query.sparse_embedding:
                search_results = EmbeddedChunk.hybrid_search(
                    query_vector=embedded_query.embedding,
                    sparse_query_vector=embedded_query.sparse_embedding,
                    limit=k // 3,
                    query_filter=None
                )
            else:
                search_results = EmbeddedChunk.search(
                    query_vector=embedded_query.embedding,
                    limit=k // 3,
                    query_filter=None
                )

        logger.info(
            f"Found {len(search_results)} chunks for query",
        )

        return search_results

    def _build_filter(self, metadata: dict | None) -> Filter | None:

        conditions = []

        if metadata.get("document_type"):
            conditions.append(
                FieldCondition(
                    key="document_type",
                    match=MatchValue(value=metadata["document_type"])
                )
            )

        if metadata.get("field"):
            conditions.append(
                FieldCondition(
                    key="field",
                    match=MatchValue(value=metadata["field"])
                )
            )

        if metadata.get("document_number"):
            conditions.append(
                FieldCondition(
                    key="document_number",
                    match=MatchValue(value=metadata["document_number"])
                )
            )

        if not conditions:
            return None

        return Filter(must=conditions)

    def rerank(self, query: str | Query, chunks: list[EmbeddedChunk], keep_top_k: int = 3) -> list[EmbeddedChunk]:
        if isinstance(query, str):
            query = Query.from_str(query)

        reranked_documents = self._reranker.generate(query=query, chunks=chunks, top_k=keep_top_k)

        logger.info(f"{len(reranked_documents)} documents reranked successfully.")

        return reranked_documents
