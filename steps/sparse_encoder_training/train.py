from typing_extensions import Annotated
from zenml import step
from loguru import logger

from llm_engineering.application.preprocessing.dispatchers import ChunkingDispatcher
from llm_engineering.application.networks import SparseEmbeddingModelSingleton
from llm_engineering.application.networks.sparse_model_utils import save_sparse_model
from llm_engineering.settings import settings


@step
def train(
    cleaned_documents: Annotated[list, "cleaned_documents"],
    algorithm: str = "bm25",
    save_path: str | None = None,
) -> Annotated[dict, "sparse_model_info"]:

    if save_path is None:
        save_path = f"models/sparse_{algorithm}_model.pkl"

    chunking_dispatcher = ChunkingDispatcher()

    logger.info(f"Chunking {len(cleaned_documents)} documents for corpus...")
    corpus = []
    for document in cleaned_documents:
        try:
            chunks = chunking_dispatcher.chunk(document)
            corpus.extend([chunk.content for chunk in chunks])
        except Exception:
            logger.exception(f"Failed to chunk document {document.id}")
            continue

    logger.info(f"Collected {len(corpus)} chunks for training")

    sparse_model = SparseEmbeddingModelSingleton(algorithm=algorithm)

    logger.info(f"Fitting {algorithm.upper()} sparse model...")
    sparse_model.fit(corpus)

    vocab_size = sparse_model.vocab_size
    logger.info(f"Training complete! Vocabulary size: {vocab_size}")

    logger.info(f"Saving model to {save_path}")
    save_sparse_model(sparse_model, save_path)

    return {
        "algorithm": algorithm,
        "vocab_size": vocab_size,
        "corpus_size": len(corpus),
        "save_path": save_path,
    }
