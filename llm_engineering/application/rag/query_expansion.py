import opik
from langchain_cohere import ChatCohere
from loguru import logger

from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings
from llm_engineering.application.rag.base import RAGStep
from llm_engineering.application.rag.prompt_templates import QueryExpansionTemplate


class QueryExpansion(RAGStep):
    @opik.track(name="QueryExpansion.generate")
    def generate(self, query: Query, expand_to_n: int) -> list[Query]:
        assert expand_to_n > 0, f"'expand_to_n' should be greater than 0. Got {expand_to_n}."

        if self._mock:
            return [query for _ in range(expand_to_n)]

        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(expand_to_n - 1)
        model = ChatCohere(
            cohere_api_key=settings.COHERE_API_KEY,
            model=settings.COHERE_MODEL_ID,
            temperature=0
        )

        chain = prompt | model

        response = chain.invoke({"question": query})
        result = response.content

        queries_content = result.strip().split(query_expansion_template.separator)

        queries = [query]
        queries += [
            query.replace_content(stripped_content)
            for content in queries_content
            if (stripped_content := content.strip())
        ]

        return queries

if __name__ == "__main__":
    query = Query.from_str("Điều 97 Bộ luật Lao động quy định về thời giờ làm việc như thế nào?")
    query_expander = QueryExpansion()
    expanded_queries = query_expander.generate(query, expand_to_n=3)
    for expanded_query in expanded_queries:
        logger.info(expanded_query.content)

