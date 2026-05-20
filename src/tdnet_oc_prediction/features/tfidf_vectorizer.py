from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer(cfg: dict) -> TfidfVectorizer:
    return TfidfVectorizer(**cfg)
