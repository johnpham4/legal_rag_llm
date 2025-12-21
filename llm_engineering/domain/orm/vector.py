from abc import ABC
from typing import Any, Callable, Dict, Generic, Type, TypeVar
from pydantic import BaseModel, Field, UUID4
import uuid
from uuid import UUID
import numpy as np
from llm_engineering.application.networks.embedding import EmbeddingModelSingleton
from loguru import logger

from qdrant_client.http import exceptions
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, Modifier, Fusion
from qdrant_client.models import PointStruct, Record, SparseVector, FusionQuery

from llm_engineering.infrastructure.db.qdrant import connection
from llm_engineering.domain.exceptions import ImproperlyConfigured

T = TypeVar("T", bound="VectorBaseDocument")

class VectorBaseDocument(BaseModel, Generic[T], ABC):
    id: UUID4 = Field(default_factory=uuid.uuid4)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id

    @classmethod
    def from_record(cls: Type[T], point: Record) -> T:
        _id = UUID(point.id, version=4)
        payload = point.payload or {}
        attributes = {
            "id": _id,
            **payload
        }
        # Check if model has embedding field and add vector if available
        if hasattr(cls, 'model_fields') and 'embedding' in cls.model_fields:
            if isinstance(point.vector, dict):
                attributes['embedding'] = point.vector.get('dense', None)
            else:
                attributes['embedding'] = point.vector or None
        return cls(**attributes)

    def to_point(self: T, **kwargs) -> PointStruct:
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)
        payload = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)

        _id = str(payload.pop("id"))
        vector = payload.pop("embedding", {})
        if vector and isinstance(vector, np.ndarray):
            vector = vector.tolist()

        sparse_embedding = payload.pop("sparse_embedding", None)

        sparse_vector = SparseVector(
            indices=sparse_embedding.get("indices", []) if sparse_embedding else [],
            values=sparse_embedding.get("values", []) if sparse_embedding else []
        )

        vectors = {
            "dense": vector,
            "text": sparse_vector
        }

        return PointStruct(id=_id, vector=vectors, payload=payload)

    def model_dump(self, **kwargs):
        dict_ = super().model_dump(**kwargs)

        dict_ = self._uuid_to_str(dict_)

        return dict_

    def _uuid_to_str(self, item: Any) -> Any:
        if isinstance(item, UUID):
            return str(item)

        if isinstance(item, list):
            return [self._uuid_to_str(v) for v in item]

        if isinstance(item, dict):
            return {k: self._uuid_to_str(v) for k, v in item.items()}

        return item

    @classmethod
    def bulk_insert(cls: Type[T], documents: list["VectorBaseDocument"]) -> bool:
        try:
            cls._bulk_insert(documents)
        except exceptions.UnexpectedResponse:
            logger.info(
                f"Collection '{cls.get_collection_name()}' does not exist. "
                "Trying to create the collection and reinsert the documents."
            )
            cls.create_collection()
            try:
                cls._bulk_insert(documents)
            except exceptions.UnexpectedResponse:
                logger.error(f"Failed to insert documents in '{cls.get_collection_name()}'.")
                return False
        return True

    @classmethod
    def create_collection(cls: Type[T]) -> bool:
        collection_name = cls.get_collection_name()
        use_vector_index = cls.get_use_vector_index()
        use_sparse_vector_index = cls.get_use_sparse_vector_index()

        if use_sparse_vector_index is True:
            vectors_config = {
                "dense": VectorParams(
                    size=EmbeddingModelSingleton().embedding_size,
                    distance=Distance.COSINE
                )
            }

            sparse_vectors_config = {"text": SparseVectorParams(modifier=Modifier.IDF)}

        elif use_vector_index is True:
            vectors_config = VectorParams(
                size=EmbeddingModelSingleton().embedding_size,
                distance=Distance.COSINE
            )
            sparse_vectors_config = None
        else:
            vectors_config = {}
            sparse_vectors_config = None

        return connection.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config
        )

    @classmethod
    def get_use_vector_index(cls: Type[T]) -> bool:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "use_vector_index"):
            return True
        return cls.Config.use_vector_index

    @classmethod
    def get_use_sparse_vector_index(cls) -> bool:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "use_sparse_vector_index"):
            return True
        return cls.Config.use_sparse_vector_index

    @classmethod
    def _bulk_insert(cls: Type[T], documents):
        points = [doc.to_point() for doc in documents]
        connection.upsert(collection_name=cls.get_collection_name(), points=points)

    @classmethod
    def get_collection_name(cls: Type[T]) -> str:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "name"):
            raise ImproperlyConfigured(
                "The class should define a Config class with the 'name' property that reflects the collection's name."
            )
        return cls.Config.name

    @classmethod
    def bulk_find(cls: Type[T], limit: int = 10, **kwargs) -> tuple[list[T], UUID | None]:
        try:
            documents, next_offset = cls._bulk_find(limit=limit, **kwargs)
        except exceptions.UnexpectedResponse:
            logger.error(f"Failed to search documents in '{cls.get_collection_name()}'.")
            documents, next_offset = [], None
        return documents, next_offset

    @classmethod
    def _bulk_find(cls: Type[T], limit: int = 10, **kwargs) -> tuple[list[T], UUID | None]:
        collection_name = cls.get_collection_name()
        offset = kwargs.pop("offset", None)
        offset = str(offset) if offset else None

        # Check if this model has embedding field to determine if we need vectors
        needs_vectors = hasattr(cls, 'model_fields') and 'embedding' in cls.model_fields

        records, next_offset = connection.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=kwargs.pop("with_payload", True),
            with_vectors=kwargs.pop("with_vectors", needs_vectors),
            offset=offset,
            **kwargs,
        )
        documents = [cls.from_record(record) for record in records]
        if next_offset is not None:
            next_offset = UUID(next_offset, version=4)
        return documents, next_offset

    @classmethod
    def search(cls: Type[T], query_vector: list, limit: int = 10, **kwargs) -> list[T]:
        try:
            documents = cls._search(query_vector=query_vector, limit=limit, **kwargs)
        except exceptions.UnexpectedResponse:
            logger.error(f"Failed to search documents in '{cls.get_collection_name()}'.")
            documents = []
        return documents

    @classmethod
    def hybrid_search(cls: Type[T], query_vector: list, sparse_query_vector: dict, limit: int = 10, **kwargs) -> list[T]:
        try:
            documents = cls._hybrid_search(
                query_vector=query_vector,
                sparse_query_vector=sparse_query_vector,
                limit=limit,
                **kwargs
            )
        except exceptions.UnexpectedResponse as e:
            logger.error(f"Failed to hybrid search in '{cls.get_collection_name()}': {str(e)}")
            documents = []
        except Exception as e:
            logger.error(f"Unexpected error in hybrid search '{cls.get_collection_name()}': {type(e).__name__}: {str(e)}")
            documents = []
        return documents

    @classmethod
    def _search(cls: Type[T], query_vector: list, limit: int = 10, **kwargs) -> list[T]:
        collection_name = cls.get_collection_name()
        needs_vectors = hasattr(cls, 'model_fields') and 'embedding' in cls.model_fields
        use_sparse = cls.get_use_sparse_vector_index()

        query_filter = kwargs.pop("query_filter", None)

        # When collection has sparse vectors, specify "dense" vector name
        query_params = {
            "collection_name": collection_name,
            "query": query_vector,
            "limit": limit,
            "with_payload": kwargs.pop("with_payload", True),
            "with_vectors": kwargs.pop("with_vectors", needs_vectors),
        }

        if use_sparse:
            query_params["using"] = "dense"

        if query_filter is not None:
            query_params["filter"] = query_filter

        records = connection.query_points(**query_params, **kwargs).points

        documents = [cls.from_record(record) for record in records]
        return documents

    @classmethod
    def _hybrid_search(cls: Type[T], query_vector: list, sparse_query_vector: dict, limit: int = 10, **kwargs) -> list[T]:
        from qdrant_client.models import Prefetch

        collection_name = cls.get_collection_name()
        needs_vectors = hasattr(cls, 'model_fields') and 'embedding' in cls.model_fields
        query_filter = kwargs.pop("query_filter", None)

        records = connection.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query_vector["indices"],
                        values=sparse_query_vector["values"]
                    ),
                    using="text",
                    limit=limit,
                    filter=query_filter
                ),
                Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=limit,
                    filter=query_filter
                )
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=kwargs.pop("with_payload", True),
            with_vectors=kwargs.pop("with_vectors", needs_vectors),
        ).points

        documents = [cls.from_record(record) for record in records]
        return documents


    @classmethod
    def group_by_class(cls: Type[T], documents: list[T]) -> Dict[T, list[T]]:
        return cls._group_by(documents, selector=lambda doc: doc.__class__)

    @classmethod
    def _group_by(cls: Type[T], documents: list[T], selector: Callable[[T], Any]) -> Dict[Any, list[T]]:
        grouped = {}
        for doc in documents:
            key = selector(doc)

            if key not in grouped:
                grouped[key] = []

            grouped[key].append(doc)

        return grouped

    @classmethod
    def get_category(cls) -> str:
        if not hasattr(cls, "Config") or not hasattr(cls.Config, "category"):
            raise ImproperlyConfigured(
                "The class should define a Config class with"
                "the 'category' property that reflects the collection's data category."
            )

        return cls.Config.category

