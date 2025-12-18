from zenml import pipeline
import steps.sparse_encoder_training.train as sparse_steps


@pipeline
def train_sparse_model(
    query_limit: int | None = None,
    algorithm: str = "bm25",
    save_path: str | None = None,
) -> None:

    raw_documents = sparse_steps.query_data_warehouse(query_limit=query_limit)

    cleaned_documents = sparse_steps.clean_documents(raw_documents)

    model_info = sparse_steps.train(
        cleaned_documents,
        algorithm=algorithm,
        save_path=save_path,
    )

    return model_info
