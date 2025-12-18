"""Preprocessing dispatchers for legal documents
Simplified version without Factory pattern since we only have one document type.
"""
from loguru import logger

from llm_engineering.domain.documents import Document
from llm_engineering.domain.cleaned_documents import CleanedDocument
from llm_engineering.domain.chunks import Chunk
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.domain.queries import Query, EmbeddedQuery

from .cleaning_data_handler import LegalCleaningHandler
from .chunking_data_handler import LegalChunkingHandler
from .embedding_data_handler import LegalEmbeddingHandler, QueryEmbeddingHandler


class CleaningDispatcher:

    def __init__(self):
        self._handler = LegalCleaningHandler()

    def clean(self, document: Document) -> CleanedDocument:
        cleaned = self._handler.clean(document)
        logger.info(
            "Document cleaned",
            doc_id=str(document.id),
            length=len(cleaned.content)
        )
        return cleaned


class ChunkingDispatcher:

    def __init__(self):
        self._handler = LegalChunkingHandler()

    def chunk(self, document: CleanedDocument) -> list[Chunk]:
        chunks = self._handler.chunk(document)
        logger.info(
            "Document chunked",
            doc_id=str(document.id),
            num_chunks=len(chunks)
        )
        return chunks


class EmbeddingDispatcher:

    def __init__(self):
        self._legal_handler = LegalEmbeddingHandler()
        self._query_handler = QueryEmbeddingHandler()

    def embed_chunks(self, chunks: list[Chunk], use_sparse: bool = True) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        embedded = self._legal_handler.embed_batch(chunks, use_sparse=use_sparse)
        logger.info("Chunks embedded", num=len(chunks))
        return embedded

    def embed_query(self, query: Query, use_sparse: bool = True) -> EmbeddedQuery:
        embedded = self._query_handler.embed(query, use_sparse=use_sparse)
        logger.info("Query embedded", query_id=str(query.id))
        return embedded

    def embed_queries(self, queries: list[Query], use_sparse: bool = True) -> list[EmbeddedQuery]:
        if not queries:
            return []

        embedded = self._query_handler.embed_batch(queries, use_sparse=use_sparse)
        logger.info("Queries embedded", num=len(queries))
        return embedded
