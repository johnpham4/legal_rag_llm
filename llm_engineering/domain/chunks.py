from abc import ABC
from pydantic import UUID4, Field
from typing import Optional

from .types import DataCategory
from .orm import VectorBaseDocument


class Chunk(VectorBaseDocument):
    content: str
    document_id: str
    document_number: str
    document_type: str
    link: str
    field: str
    platform: str = "thuvienphapluat.vn"

    class Config:
        name = "chunked_documents"



