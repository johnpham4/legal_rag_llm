from abc import ABC, abstractmethod
from typing import Generic, TypeVar, cast

from llm_engineering.application.networks import EmbeddingModelSingleton
from llm_engineering.domain.chunks import Chunk
from llm_engineering.domain.chunks import Chunk
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.domain.queries import EmbeddedQuery, Query

ChunkT = TypeVar("ChunkT", bound=Chunk)
EmbeddedChunkT = TypeVar("EmbeddedChunkT", bound=EmbeddedChunk)

embedding_model = EmbeddingModelSingleton()


class EmbeddingDataHandler(ABC, Generic[ChunkT, EmbeddedChunkT]):

    def embed(self, data_model: ChunkT) -> EmbeddedChunkT:
        return self.embed_batch([data_model])[0]

    def embed_batch(self, data_models: list[ChunkT]) -> list[EmbeddedChunkT]:
        embedding_model_input = [model.content for model in data_models]
        embeddings = embedding_model(embedding_model_input, to_list=True)

        embedded_chunks = [
            self.map_model(data_model, cast(list[float], embedding))
            for data_model, embedding in zip(data_models, embeddings, strict=False)
        ]

        return embedded_chunks

    @abstractmethod
    def map_model(self, data_model: ChunkT, embedding: list[float]) -> EmbeddedChunkT:
        pass


class QueryEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: Query, embedding: list[float]) -> EmbeddedQuery:
        return EmbeddedQuery(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


class LegalEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: Chunk, embedding: list[float]) -> EmbeddedChunk:
        return EmbeddedChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
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
