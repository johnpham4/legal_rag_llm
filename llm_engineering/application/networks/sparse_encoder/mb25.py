import re
import math
import pickle
from pathlib import Path
from collections import Counter
from functools import lru_cache
from tqdm import tqdm
from loguru import logger

from .base import BaseSparseEncoder
from llm_engineering.application.networks.base import SingletonMeta
from llm_engineering.settings import settings


class BM25SparseEncoder(BaseSparseEncoder, metaclass=SingletonMeta):

    def __init__(self, max_terms: int = 128, k1: float = 1.5, b: float = 0.75):
        # Singleton only calls __init__ once - if already initialized, return
        if hasattr(self, '_is_fitted'):
            return

        # Initialize attributes
        self.vocab = {}
        self.idf = {}
        self.avgdl = 0.0
        self.max_terms = max_terms
        self.k1 = k1
        self.b = b

        self.model_path = settings.SPARSE_MODEL_PATH
        self._is_fitted = False

        if Path(self.model_path).exists():
            try:
                self._load_from_path(self.model_path)
                self._is_fitted = True
                logger.info(f"BM25SparseEncoder loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load BM25 model: {e}")
                self._is_fitted = False
        else:
            logger.warning(f"Model file not found: {self.model_path}")

    @staticmethod
    @lru_cache(maxsize=10000)
    def _tokenize(text: str) -> tuple:
        pattern = r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+'
        tokens = re.findall(pattern, text.lower())
        return tuple(tokens)

    def fit(self, corpus: list[str]) -> None:
        df = Counter()
        N = len(corpus)
        total_doc_len = 0

        for text in tqdm(corpus, desc="Building BM25 vocabulary"):
            tokens = self._tokenize(text)
            total_doc_len += len(tokens)

            unique_tokens = set(tokens)  # Unique tokens per document
            for term in unique_tokens:
                df[term] += 1

        self.avgdl = total_doc_len / N if N > 0 else 0.0

        sorted_terms = sorted(df.keys())
        self.vocab = {term: idx for idx, term in enumerate(sorted_terms)}

        self.idf = {
            term: math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            for term in df
        }

        self._is_fitted = True


    def encode(self, input_text: str | list[str]) -> dict | list[dict]:
        if not self._is_fitted:
            raise ValueError("Encoder must be fitted before encoding. Call fit() or load() first.")

        if isinstance(input_text, list):
            return [self._encode(text) for text in input_text]
        else:
            return self._encode(input_text)

    def _encode(self, text: str) -> dict:
        if not text:
            return {"indices": [], "values": []}

        tokens = self._tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)

        scores = {}
        for term, freq in tf.items():
            if term in self.vocab:
                idf = self.idf[term]

                # Length normalization factor
                norm_factor = 1.0 - self.b + self.b * (doc_len / self.avgdl) if self.avgdl > 0 else 1.0

                # BM25 score = IDF * (freq * (k1 + 1)) / (freq + k1 * norm)
                bm25_score = idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * norm_factor)
                scores[term] = bm25_score
            else:
                continue

        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.max_terms]

        return {
            "indices": [self.vocab[term] for term, _ in top_items],
            "values": [score for _, score in top_items]
        }

    def save(self, model_path: str) -> bool:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        state = {
            "vocab": self.vocab,
            "idf": self.idf,
            "avgdl": self.avgdl,
            "max_terms": self.max_terms,
            "k1": self.k1,
            "b": self.b
        }

        with open(model_path, "wb") as f:
            pickle.dump(state, f)

        return True

    def _load_from_path(self, model_path: str) -> None:
        """Internal method to load state into current instance."""
        with open(model_path, "rb") as f:
            state = pickle.load(f)

        self.vocab = state["vocab"]
        self.idf = state["idf"]
        self.avgdl = state["avgdl"]
        self.max_terms = state.get("max_terms", self.max_terms)
        self.k1 = state.get("k1", self.k1)
        self.b = state.get("b", self.b)
        self._is_fitted = True

    @classmethod
    def load(cls, model_path: str):
        """Load and return a new instance (deprecated, use __init__ with model_path instead)."""
        with open(model_path, "rb") as f:
            state = pickle.load(f)

        instance = cls(
            max_terms=state["max_terms"],
            k1=state["k1"],
            b=state["b"]
        )
        instance.vocab = state["vocab"]
        instance.idf = state["idf"]
        instance.avgdl = state["avgdl"]
        instance._is_fitted = True

        return instance

    @classmethod
    def algorithm(cls) -> "BM25SparseEncoder":
        return cls.__class__
