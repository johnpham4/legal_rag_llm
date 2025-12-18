import re
import math
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache
from tqdm import tqdm

from .base import BaseSparseEncoder


class TFIDFEncoder(BaseSparseEncoder):

    def __init__(self, max_terms: int = 128):
        self.vocab = {}  # term -> index mapping
        self.idf = {}    # term -> IDF score
        self.max_terms = max_terms
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

        # Step 1: Count document frequency
        for text in tqdm(corpus, desc="Building TF-IDF vocabulary"):
            tokens = set(self._tokenize(text))  # Unique tokens per document
            for term in tokens:
                df[term] += 1

        # Step 2: Build vocabulary (sorted for consistent indexing)
        sorted_terms = sorted(df.keys())
        self.vocab = {term: idx for idx, term in enumerate(sorted_terms)}

        # Step 3: Calculate IDF for each term
        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf = {
            term: math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            for term in df
        }

        self._is_fitted = True

    def encode(self, input_text: str | list[str]) -> dict | list[dict]:

        if not self._is_fitted:
            raise ValueError("Encoder must be fitted before encoding. Call fit() first.")

        if isinstance(input_text , list):
            encode_list = []

            for text in input_text:
                dict_ = self._encode(text)
                encode_list.append(dict_)

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
        state = {"vocab": self.vocab, "idf": self.idf, "max_terms": self.max_terms}
        with open(model_path, "wb") as f:
            pickle.dump(state, f)
        return True

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
    def algorithm(cls) -> "TFIDFEncoder":
        return cls.__class__