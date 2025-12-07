import hashlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from llm_engineering.domain.chunks import Chunk
from llm_engineering.domain.cleaned_documents import (
    CleanedDocument,
)

from .operations import chunk_legal_document

CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)
ChunkT = TypeVar("ChunkT", bound=Chunk)


class ChunkingDataHandler(ABC, Generic[CleanedDocumentT, ChunkT]):
    """
    Abstract class for all Chunking data handlers.
    All data transformations logic for the chunking step is done here
    """

    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

    @abstractmethod
    def chunk(self, data_model: CleanedDocumentT) -> list[ChunkT]:
        pass


class LegalChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "min_length": 100,
            "max_length": 1000,
            "chunking_strategy": "semantic",
        }

    def chunk(self, data_model: CleanedDocument) -> list[Chunk]:
        data_models_list = []
        cleaned_content = data_model.content

        chunks = chunk_legal_document(
            cleaned_content,
            min_length=self.metadata["min_length"],
            max_length=self.metadata["max_length"]
        )

        for chunk_index, chunk_content in enumerate(chunks):
            chunk_id_str = f"{data_model.id}-{chunk_index}"
            chunk_id = hashlib.md5(chunk_id_str.encode()).hexdigest()

            model = Chunk(
                id=UUID(chunk_id, version=4),
                content=chunk_content,
                document_id=str(data_model.id),
                document_number=data_model.document_number,
                document_type=data_model.document_type,
                link=data_model.link,
                field=data_model.field,
                platform=data_model.platform,
            )
            data_models_list.append(model)

        return data_models_list

