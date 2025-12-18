from typing import Optional

from .types import DataCategory
from .orm import VectorBaseDocument


class CleanedDocument(VectorBaseDocument):
    """Vietnamese legal document model"""
    content: str
    document_number: str
    document_type: str
    title: Optional[str] = None
    link: str
    field: str
    platform: str = "thuvienphapluat.vn"

    class Config:
        name = "cleaned_documents"
        use_vector_index = False
        use_sparse_vector_index = False


