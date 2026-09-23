import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

DEFAULT_VECTORIZER_PATH = Path(__file__).resolve().parent.parent / 'models' / 'tfidf_vectorizer.pkl'


def create_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    min_df: int = 1,
    max_df: float = 0.95,
    stop_words: str = 'english',
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
        lowercase=True,
        sublinear_tf=True,
    )


def fit_vectorizer(messages: list[str], save_path: str | Path | None = None) -> TfidfVectorizer:
    vectorizer = create_vectorizer()
    vectorizer.fit(messages)
    path = Path(save_path) if save_path else DEFAULT_VECTORIZER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    return vectorizer


def load_vectorizer(path: str | Path | None = None) -> TfidfVectorizer:
    p = Path(path) if path else DEFAULT_VECTORIZER_PATH
    if not p.exists():
        raise FileNotFoundError(f"Vectorizer not found at {p}. Run training first.")
    return joblib.load(p)


def transform(messages: list[str], vectorizer: TfidfVectorizer | None = None) -> "scipy.sparse":
    if vectorizer is None:
        vectorizer = load_vectorizer()
    return vectorizer.transform(messages)


if __name__ == '__main__':
    samples = [
        "block replicated nodes successfully",
        "connection timeout host",
        "authentication failed user ip",
    ]
    v = fit_vectorizer(samples)
    X = transform(samples, v)
    print("Shape:", X.shape)
