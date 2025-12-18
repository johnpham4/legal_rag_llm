from llm_engineering.infrastructure.llm.base import BaseLLMClient


class LlamaClient(BaseLLMClient):
    def __init__(self):
        pass

    def generate(self, prompt, temperature = 0.3, **kwargs):
        return super().generate(prompt, temperature, **kwargs)