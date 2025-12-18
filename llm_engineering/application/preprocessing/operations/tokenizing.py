from underthesea import word_tokenize
import re

def tokenize_vi(text: str) -> list[str]:
    text = text.strip().lower()

    tokens = word_tokenize(text, format="text").split()
    return [t for t in tokens if len(t) > 1]
