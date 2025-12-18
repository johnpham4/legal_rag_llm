from collections import Counter
from loguru import logger

from .base import SingletonMeta
from .sparse_encoder import BM25Encoder, TFIDFEncoder


class SparseEmbeddingModelSingleton(metaclass=SingletonMeta):

    def __init__(self, algorithm: str = "bm25", **kwargs):

        if algorithm.lower() == "bm25":
            self._encoder = BM25Encoder(
                max_terms=kwargs.get("max_terms", 128),
                k1=kwargs.get("k1", 1.5),
                b=kwargs.get("b", 0.75)
            )
        elif algorithm.lower() == "tfidf":
            self._encoder = TFIDFEncoder(
                max_terms=kwargs.get("max_terms", 128)
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'bm25' or 'tfidf'")

        self._algorithm = algorithm.lower()
        self._is_fitted = False

    def fit(self, corpus: list[str]) -> None:

        if corpus:
            self._encoder.fit(corpus)
            self._is_fitted = True


    def load(self, model_path: str):
        return self._encoder.load(model_path=model_path)

    def save(self, model_path: str) -> bool:
        return self._encoder.save(model_path=model_path)

    def __call__(self, input_text: str | list[str], to_list: bool=True) -> dict | list[dict]:
        try:
            dict_ = self._encoder.encode(input_text)
        except Exception:
            logger.error(f"Error generating embeddings for {self._model_id=} and {input_text=}")

            return [{}] if to_list else {}

        if to_list:
            dict_ = [d for d in dict_]

        return dict_

    @property
    def algorithm(self) -> str:
        return self._algorithm

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def vocab_size(self) -> int:
        if hasattr(self._encoder, 'vocab'):
            return len(self._encoder.vocab)
        return 0
