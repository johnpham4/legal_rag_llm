from typing import Optional

from .orm import NoSQLBaseDocument


class Document(NoSQLBaseDocument):
    """Vietnamese legal document model"""
    content: str
    document_number: str
    document_type: str
    title: Optional[str] = None
    link: str
    field: str
    platform: str = "thuvienphapluat.vn"

    class Settings:
        name = "legal_documents"


