from abc import ABC, abstractmethod


class BaseLLMClient(ABC):

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        **kwargs
    ) -> str:
        pass
