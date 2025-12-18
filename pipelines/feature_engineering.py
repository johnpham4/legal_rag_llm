from zenml import pipeline
import steps.feature_engineering as fe_steps

@pipeline
def feature_engineering(
    query_limit: int | None = None,
    batch_size: int = 10,
    use_sparse: bool = True,
) -> None:
    """Feature engineering pipeline for Vietnamese legal documents.

    Args:
        query_limit: Maximum number of documents to process (None = all)
        batch_size: Number of chunks to embed per batch
        use_sparse: Whether to use sparse embeddings (BM25) for hybrid search
    """
    # Step 1: Query raw documents from MongoDB
    raw_documents = fe_steps.query_data_warehouse(query_limit=query_limit)

    # Step 2: Clean documents (remove signatures, normalize structure)
    cleaned_documents = fe_steps.clean_documents(raw_documents)

    # Step 3: Chunk and embed documents (semantic + sparse if enabled)
    embedded_documents = fe_steps.chunk_and_embed(
        cleaned_documents,
        batch_size=batch_size,
        use_sparse=use_sparse,
    )

    # Step 4: Load embedded chunks to vector DB (for semantic/hybrid retrieval)
    last_step = fe_steps.load_to_vector_db(embedded_documents)

    return last_step.invocation_id