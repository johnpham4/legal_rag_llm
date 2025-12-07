from loguru import logger
from langchain_cohere import ChatCohere

from llm_engineering.infrastructure.llm.base import BaseLLMClient
from llm_engineering.settings import settings


class CohereLLMClient(BaseLLMClient):
    """Cohere LLM client using LangChain wrapper."""

    def __init__(self):
        self._client = ChatCohere(
            cohere_api_key=settings.COHERE_API_KEY,
            model=settings.COHERE_MODEL_ID,
        )
        logger.info(f"Initialized Cohere client with model: {settings.COHERE_MODEL_ID}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        **kwargs
    ) -> str:
        try:
            self._client.temperature = temperature
            response = self._client.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Cohere generation failed: {e}")
            raise
