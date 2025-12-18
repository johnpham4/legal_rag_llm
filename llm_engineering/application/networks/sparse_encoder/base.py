from abc import ABC, abstractmethod

class BaseSparseEncoder(ABC):
    @abstractmethod
    def fit(self, corpus: list[str]) -> None: ...

    @abstractmethod
    def encode(self, text: str) -> dict: ...

    @abstractmethod
    def load(self, model_path: str) -> bool: ...

    @classmethod
    @abstractmethod
    def save(self, model_path: str) -> bool: ...