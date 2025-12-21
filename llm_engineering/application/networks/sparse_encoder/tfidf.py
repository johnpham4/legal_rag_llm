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

class TFIDFSparseEncoder(BaseSparseEncoder, metaclass=SingletonMeta):

    def __init__(self, max_terms: int = 128):
        # Singleton only calls __init__ once - if already initialized, return
        if hasattr(self, '_is_fitted'):
            return

        # Initialize attributes
        self.vocab = {}
        self.idf = {}
        self.max_terms = max_terms
        self.model_path = settings.SPARSE_MODEL_PATH
        self._is_fitted = False

        if Path(self.model_path).exists():
            try:
                self._load_from_path(self.model_path)
                self._is_fitted = True
                logger.info(f"TFIDFSparseEncoder loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load TFIDF model: {e}")
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

        for text in tqdm(corpus, desc="Building TF-IDF vocabulary"):
            tokens = set(self._tokenize(text))
            for term in tokens:
                df[term] += 1

        sorted_terms = sorted(df.keys())
        self.vocab = {term: idx for idx, term in enumerate(sorted_terms)}

        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
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
            return self._encode(text=input_text)



    def _encode(self, text):
        tokens = self._tokenize(text)
        tf = Counter(tokens)

        scores = {}
        for term, freq in tf.items():
            if term in self.vocab:
                # TF-IDF = TF * IDF
                tfidf_score = freq * self.idf[term]
                scores[term] = tfidf_score

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
            "max_terms": self.max_terms
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
        self.max_terms = state.get("max_terms", self.max_terms)
        self._is_fitted = True

    @classmethod
    def load(cls, model_path: str):
        with open(model_path, "rb") as f:
            state = pickle.load(f)

        instance = cls(max_terms=state["max_terms"])
        instance.vocab = state["vocab"]
        instance.idf = state["idf"]
        instance._is_fitted = True

        return instance

    @classmethod
    def algorithm(cls) -> "TFIDFSparseEncoder":
        return cls.__class__