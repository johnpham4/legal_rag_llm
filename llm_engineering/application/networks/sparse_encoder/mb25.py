import re
import math
import pickle
from pathlib import Path
from collections import Counter
from functools import lru_cache

from .base import BaseSparseEncoder


class BM25Encoder(BaseSparseEncoder):

    def __init__(self, max_terms: int = 128, k1: float = 1.5, b: float = 0.75):
        self.vocab = {}    # term -> index mapping
        self.idf = {}      # term -> IDF score
        self.avgdl = 0.0   # Average document length
        self.max_terms = max_terms
        self.k1 = k1
        self.b = b
        self._is_fitted = False

    @staticmethod
    @lru_cache(maxsize=10000)
    def _tokenize(text: str) -> tuple:
        pattern = r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+'
        tokens = re.findall(pattern, text.lower())
        return tuple(tokens)

    def fit(self, corpus: list[str]) -> None:
        df = Counter()  # Document frequency for each term
        N = len(corpus)  # Number of documents
        total_doc_len = 0  # Sum of all document lengths

        # Step 1: Count document frequency and total length
        for text in corpus:
            tokens = self._tokenize(text)
            total_doc_len += len(tokens)

            unique_tokens = set(tokens)  # Unique tokens per document
            for term in unique_tokens:
                df[term] += 1

        # Step 2: Calculate average document length
        self.avgdl = total_doc_len / N if N > 0 else 0.0

        # Step 3: Build vocabulary (sorted for consistent indexing)
        sorted_terms = sorted(df.keys())
        self.vocab = {term: idx for idx, term in enumerate(sorted_terms)}

        # Step 4: Calculate IDF for each term
        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf = {
            term: math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            for term in df
        }

        self._is_fitted = True

    def encode(self, text: str) -> dict:
        if not text:
            return {"indices": [], "values": []}

        if not self._is_fitted:
            raise ValueError("Encoder must be fitted before encoding. Call fit() first.")

        # Step 1: Tokenize and count term frequency
        tokens = self._tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)

        # Step 2: Calculate BM25 score for each term
        scores = {}
        for term, freq in tf.items():
            if term in self.vocab:
                idf = self.idf[term]

                # Length normalization factor
                # norm = 1 - b + b * (doc_len / avgdl)
                norm_factor = 1.0 - self.b + self.b * (doc_len / self.avgdl) if self.avgdl > 0 else 1.0

                # BM25 score = IDF * (freq * (k1 + 1)) / (freq + k1 * norm)
                bm25_score = idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * norm_factor)
                scores[term] = bm25_score

        # Step 3: Keep only top-K terms by score
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.max_terms]

        # Step 4: Convert to indices and values
        return {
            "indices": [self.vocab[term] for term, _ in top_items],
            "values": [score for _, score in top_items]
        }

    def save(self, model_path: str) -> bool:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        state = {"vocab": self.vocab, "idf": self.idf, "avgdl": self.avgdl,
                 "max_terms": self.max_terms, "k1": self.k1, "b": self.b}
        with open(model_path, "wb") as f:
            pickle.dump(state, f)
        return True

    def load(self, model_path: str):
        with open(model_path, "rb") as f:
            state = pickle.load(f)
        instance = self(max_terms=state["max_terms"], k1=state["k1"], b=state["b"])
        instance.vocab = state["vocab"]
        instance.idf = state["idf"]
        instance.avgdl = state["avgdl"]
        instance._is_fitted = True
        return instance
