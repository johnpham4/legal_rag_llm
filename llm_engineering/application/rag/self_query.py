import json
from langchain_cohere import ChatCohere
from loguru import logger

from llm_engineering.domain.queries import Query
from llm_engineering.domain.types import LegalField, DocumentType
from llm_engineering.settings import settings
from llm_engineering.application.rag.base import RAGStep
from llm_engineering.application.rag.prompt_templates import SelfQueryTemplate

class SelfQuery(RAGStep):
    def generate(self, query: Query) -> Query:
        if self._mock:
            return query

        try:
            prompt = SelfQueryTemplate().create_template()
            model = ChatCohere(
                cohere_api_key=settings.COHERE_API_KEY,
                model=settings.COHERE_MODEL_ID,
                temperature=0
            )

            chain = prompt | model
            response = chain.invoke({"question": query.content})
            response_text = response.content.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.strip("`").removeprefix("json").strip()

            metadata_dict = json.loads(response_text)

            # Only keep 3 fields that are actually used for filtering
            filtered_metadata = {}

            # Validate and add document_type
            if metadata_dict.get("document_type"):
                doc_type = metadata_dict["document_type"]
                valid_types = [t.value for t in DocumentType]
                if doc_type in valid_types:
                    filtered_metadata["document_type"] = doc_type
                else:
                    logger.warning(f"Invalid document_type '{doc_type}'. Valid: {valid_types}")

            # Validate and add field
            if metadata_dict.get("field"):
                field_value = metadata_dict["field"]
                valid_fields = [f.value for f in LegalField]
                if field_value in valid_fields:
                    filtered_metadata["field"] = field_value
                else:
                    logger.warning(f"Invalid field '{field_value}'. Valid: {valid_fields}")

            # Add document_number (no validation needed)
            if metadata_dict.get("document_number"):
                filtered_metadata["document_number"] = metadata_dict["document_number"]

            # Update query metadata with only validated, used fields
            query.metadata.update(filtered_metadata)

            logger.info(f"Extracted metadata: {filtered_metadata}")

            return query

        except Exception as e:
            logger.error(f"Self-query failed: {e}")
            return query



if __name__ == "__main__":
    query = Query.from_str("Điều 97 Bộ luật Lao động quy định về thời giờ làm việc như thế nào?")
    self_query = SelfQuery()
    query = self_query.generate(query)

    logger.info(f"Query content: {query.content}")
    logger.info(f"Extracted self query: {json.loads(query.metadata)}")
