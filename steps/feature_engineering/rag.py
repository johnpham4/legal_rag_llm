from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application import utils
from llm_engineering.application.preprocessing.dispatchers import ChunkingDispatcher, EmbeddingDispatcher
from llm_engineering.domain.chunks import Chunk
from llm_engineering.domain.embedded_chunks import EmbeddedChunk


@step
def chunk_and_embed(
    cleaned_documents: Annotated[list, "cleaned_documents"],
    batch_size: int = 10,
) -> Annotated[list, "embedded_documents"]:
    """Chunk documents and generate embeddings.

    Args:
        cleaned_documents: List of cleaned documents
        batch_size: Number of chunks per embedding batch
    """
    from loguru import logger

    # Initialize dispatchers
    chunking_dispatcher = ChunkingDispatcher()
    embedding_dispatcher = EmbeddingDispatcher()

    metadata = {
        "chunking": {},
        "embedding": {"batch_size": batch_size},
        "num_documents": len(cleaned_documents),
        "failed_documents": 0,
    }

    embedded_chunks = []
    for document in cleaned_documents:
        try:
            chunks = chunking_dispatcher.chunk(document)
            metadata["chunking"] = _add_chunks_metadata(chunks, metadata["chunking"])

            for batched_chunks in utils.misc.batch(chunks, batch_size):
                batched_embedded_chunks = embedding_dispatcher.embed_chunks(batched_chunks)
                embedded_chunks.extend(batched_embedded_chunks)
        except Exception:
            logger.exception(f"Failed to process document {document.id}")
            metadata["failed_documents"] += 1
            continue

    metadata["embedding"] = _add_embeddings_metadata(embedded_chunks, metadata["embedding"])
    metadata["num_chunks"] = len(embedded_chunks)
    metadata["num_embedded_chunks"] = len(embedded_chunks)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="embedded_documents", metadata=metadata)

    return embedded_chunks


def _add_chunks_metadata(chunks: list[Chunk], metadata: dict) -> dict:
    for chunk in chunks:
        chunk_info = {
            "document_id": chunk.document_id,
        }

        if "chunks" not in metadata:
            metadata["chunks"] = []
        metadata["chunks"].append(chunk_info)
        metadata["total_chunks"] = metadata.get("total_chunks", 0) + 1

    return metadata


def _add_embeddings_metadata(embedded_chunks: list[EmbeddedChunk], metadata: dict) -> dict:
    for embedded_chunk in embedded_chunks:
        if "embedding_model" not in metadata and hasattr(embedded_chunk, "metadata"):
            metadata["embedding_model"] = getattr(embedded_chunk, "metadata", {})

        metadata["total_embedded"] = metadata.get("total_embedded", 0) + 1

    return metadata
