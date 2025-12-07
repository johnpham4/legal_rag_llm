from zenml import step, get_step_context
from typing_extensions import Annotated

from llm_engineering.application.preprocessing.dispatchers import CleaningDispatcher
from llm_engineering.domain.cleaned_documents import CleanedDocument

@step
def clean_documents(
    documents: Annotated[list, "raw_documents"],
) -> Annotated[list, "cleaned_documents"]:
    dispatcher = CleaningDispatcher()
    cleaned_documents = [dispatcher.clean(doc) for doc in documents]

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="cleaned_documents", metadata=_get_metadata(cleaned_documents))

    return cleaned_documents

def _get_metadata(cleaned_documents: list[CleanedDocument]) -> dict:
    metadata = {
        "num_documents": len(cleaned_documents)
    }
    for document in cleaned_documents:
        _type = document.document_type
        if _type not in metadata:
            metadata[_type] = {}
        metadata[_type]["num_documents"] = metadata[_type].get("num_documents", 0) + 1

    return metadata
