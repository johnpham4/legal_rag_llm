from abc import ABC, abstractmethod
from typing import TypeVar, Type, Generic

from llm_engineering.domain.cleaned_documents import CleanedDocument
from llm_engineering.domain.documents import Document
from llm_engineering.application.preprocessing.operations.cleaning import clean_legal_text

DocumentT = TypeVar("DocumentT", bound=Document)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)

class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):

    @abstractmethod
    def clean(self, data_model: DocumentT) -> CleanedDocumentT:
        pass


class LegalCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: Document) -> CleanedDocument:
        return CleanedDocument(
            id=data_model.id,
            title=data_model.title,
            content=clean_legal_text(data_model.content),
            document_number=data_model.document_number,
            document_type=data_model.document_type,
            link=data_model.link,
            field=data_model.field,
            platform=data_model.platform,
        )
