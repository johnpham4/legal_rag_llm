from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.infrastructure.llm.cohere_client import CohereLLMClient


class CohereInference:
    def __init__(self, mock: bool = False):
        self.retriever = ContextRetriever(mock=mock)
        self.llm = CohereLLMClient()

    def execute(self, query: str, k: int = 3, temperature: float = 0.3) -> dict:
        documents = self.retriever.search(query, k=k)
        context = EmbeddedChunk.to_context(documents)

        prompt = f"""Dựa vào ngữ cảnh sau để trả lời câu hỏi.

        Ngữ cảnh:
        {context}

        Câu hỏi: {query}

        Trả lời:"""

        answer = self.llm.generate(prompt, temperature=temperature)

        sources = [
            {
                "document_id": doc.document_id,
                "document_type": doc.document_type,
                "field": doc.field,
                "document_number": doc.document_number,
                "content_preview": doc.content[:200] if doc.content else ""
            }
            for doc in documents
        ]

        return {
            "answer": answer,
            "sources": sources,
            "metadata": {"query": query, "k": k, "temperature": temperature}
        }
