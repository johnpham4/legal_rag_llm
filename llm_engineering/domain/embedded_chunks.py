from .types import DataCategory
from .orm import VectorBaseDocument


class EmbeddedChunk(VectorBaseDocument):
    content: str
    embedding: list[float] | None
    sparse_embedding: dict | None = None
    document_id: str
    document_number: str
    document_type: str
    link: str
    field: str
    platform: str = "thuvienphapluat.vn"

    class Config:
        name = "embedded_chunks"
        use_vector_index = True
        use_sparse_vector_index = True

    @classmethod
    def to_context(cls, chunks: list["EmbeddedChunk"]) -> str:
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"""
            Chunk {i + 1}:
            Type: {chunk.__class__.__name__}
            Platform: {chunk.platform}
            Type: {chunk.document_type}\n
            Content: {chunk.content}\n
            """

        return context


