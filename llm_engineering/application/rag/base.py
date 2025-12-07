from abc import ABC, abstractmethod
from typing import Any

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from llm_engineering.domain.queries import Query

class PromptTemplateFactory(BaseModel, ABC):
    @abstractmethod
    def create_template(self) -> PromptTemplate:
        pass

class RAGStep(ABC):
    def __init__(self, mock: bool = False) -> None:
        self._mock = mock

    @abstractmethod
    def generate(self, query: Query, *args, **kwargs) -> Any:
        pass