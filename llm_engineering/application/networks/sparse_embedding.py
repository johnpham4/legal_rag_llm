from .sparse_encoder import BM25SparseEncoder, TFIDFSparseEncoder


def get_sparse_encoder(algorithm: str = "bm25", **kwargs):
    algorithm = algorithm.lower()

    if algorithm == "bm25":
        return BM25SparseEncoder(
            max_terms=kwargs.get("max_terms", 128),
            k1=kwargs.get("k1", 1.5),
            b=kwargs.get("b", 0.75),
        )
    elif algorithm == "tfidf":
        return TFIDFSparseEncoder(
            max_terms=kwargs.get("max_terms", 128)
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'bm25' or 'tfidf'")


__all__ = ["get_sparse_encoder", "BM25SparseEncoder", "TFIDFSparseEncoder"]
