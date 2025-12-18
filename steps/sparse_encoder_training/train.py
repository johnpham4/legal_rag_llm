from typing_extensions import Annotated
from llm_engineering import settings
from zenml import step, get_step_context
from loguru import logger

from llm_engineering.application.preprocessing.dispatchers import ChunkingDispatcher
from llm_engineering.application.networks import SparseEmbeddingModelSingleton


@step
def train(
    cleaned_documents: Annotated[list, "cleaned_documents"],
    save_path: str | None = None,
) -> Annotated[int, "num_trained"]:

    algorithm = settings.SPARSE_ALGORITHM

    if save_path is None:
        save_path = f"models/sparse_{algorithm}_model.pkl"

    chunking_dispatcher = ChunkingDispatcher()

    logger.info(f"Chunking {len(cleaned_documents)} documents for corpus...")
    corpus = []
    failed_count = 0
    for document in cleaned_documents:
        try:
            chunks = chunking_dispatcher.chunk(document)
            corpus.extend([chunk.content for chunk in chunks])
        except Exception:
            logger.exception(f"Failed to chunk document {document.id}")
            failed_count += 1
            continue

    logger.info(f"Collected {len(corpus)} chunks for training")

    sparse_model = SparseEmbeddingModelSingleton(algorithm=algorithm)

    logger.info(f"Fitting {algorithm.upper()} sparse model...")
    sparse_model.fit(corpus)

    vocab_size = sparse_model.vocab_size
    logger.info(f"Training complete! Vocabulary size: {vocab_size}")

    logger.info(f"Saving model to {save_path}")
    sparse_model.save(save_path)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="num_trained",
        metadata={
            "algorithm": algorithm,
            "vocab_size": vocab_size,
            "corpus_size": len(corpus),
            "num_documents": len(cleaned_documents),
            "failed_documents": failed_count,
            "save_path": save_path,
        }
    )

    return len(corpus)
