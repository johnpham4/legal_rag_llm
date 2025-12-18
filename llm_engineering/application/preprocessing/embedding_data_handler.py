from abc import ABC, abstractmethod
from typing import Generic, TypeVar, cast

from llm_engineering import settings
from llm_engineering.application.networks import EmbeddingModelSingleton, SparseEmbeddingModelSingleton
from llm_engineering.domain.chunks import Chunk
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.domain.queries import EmbeddedQuery, Query


ChunkT = TypeVar("ChunkT", bound=Chunk)
EmbeddedChunkT = TypeVar("EmbeddedChunkT", bound=EmbeddedChunk)

embedding_model = EmbeddingModelSingleton()
sparse_embedding_model = SparseEmbeddingModelSingleton(algorithm=settings.SPARSE_ALGORITHM)

class EmbeddingDataHandler(ABC, Generic[ChunkT, EmbeddedChunkT]):

    def embed(self, data_model: ChunkT, use_sparse: bool = True) -> EmbeddedChunkT:
        return self.embed_batch([data_model], use_sparse=use_sparse)[0]

    def embed_batch(self, data_models: list[ChunkT], use_sparse: bool = True) -> list[EmbeddedChunkT]:
        embedding_model_input = [model.content for model in data_models]
        embeddings = embedding_model(embedding_model_input, to_list=True)

        if use_sparse:
            sparse_embeddings = []
            for text in embedding_model_input:
                sparse_emb = sparse_embedding_model(text, to_list=False)
                sparse_embeddings.append(sparse_emb)
        else:
            sparse_embeddings = [None] * len(data_models)

        embedded_chunks = [
            self.map_model(data_model, cast(list[float], embedding), sparse_emb)
            for data_model, embedding, sparse_emb in zip(data_models, embeddings, sparse_embeddings, strict=False)
        ]

        return embedded_chunks

    @abstractmethod
    def map_model(self, data_model: ChunkT, embedding: list[float], sparse_embedding: dict | None) -> EmbeddedChunkT:
        pass


class QueryEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: Query, embedding: list[float], sparse_embedding: dict | None) -> EmbeddedQuery:
        return EmbeddedQuery(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            sparse_embedding=sparse_embedding,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


class LegalEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: Chunk, embedding: list[float], sparse_embedding: dict | None) -> EmbeddedChunk:
        return EmbeddedChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            sparse_embedding=sparse_embedding,
            platform=data_model.platform,
            document_id=data_model.document_id,
            document_number=data_model.document_number,
            document_type=data_model.document_type,
            link=data_model.link,
            field=data_model.field,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )
