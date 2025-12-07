from pydantic import UUID4, Field

from llm_engineering.domain.orm import VectorBaseDocument
from llm_engineering.domain.types import DataCategory

class Query(VectorBaseDocument):
    content: str
    metadata: dict = Field(default_factory=dict)

    class Config:
        name = "queries"

    @classmethod
    def from_str(cls, query: str) -> "Query":
        return Query(content=query.strip("\n "))

    def replace_content(self, new_content: str) -> "Query":
        return Query(
            id=self.id,
            content=new_content,
            metadata=self.metadata,
        )

class EmbeddedQuery(Query):
    embedding: list[float]

    class Config:
        name = "embedded_queries"
        use_vector_index = True