"""LDA topic modeling utilities for SmartDoc."""

from pathlib import Path

import joblib
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
VECTORIZER_PATH = MODEL_DIR / "count_vectorizer.joblib"
LDA_PATH = MODEL_DIR / "lda_model.joblib"


def build_topic_model(
    documents,
    n_topics: int = 8,
    max_features: int = 10000,
    min_df: int = 5,
    max_df: float = 0.95,
    random_state: int = 42,
):
    """Fit a CountVectorizer + LDA model and return both objects."""
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    document_term_matrix = vectorizer.fit_transform(documents)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
        max_iter=10,
        n_jobs=-1,
    )
    lda.fit(document_term_matrix)
    return vectorizer, lda


def get_document_topic_distribution(documents, vectorizer, lda):
    """Transform documents into their LDA topic distributions."""
    matrix = vectorizer.transform(documents)
    return lda.transform(matrix)


def get_top_words_per_topic(vectorizer, lda, n_words: int = 10):
    """Return the most important words for every discovered topic."""
    feature_names = vectorizer.get_feature_names_out()
    topics = {}

    for topic_index, topic_weights in enumerate(lda.components_, start=1):
        top_indices = topic_weights.argsort()[-n_words:][::-1]
        topics[topic_index] = [feature_names[index] for index in top_indices]

    return topics


def save_topic_model(vectorizer, lda):
    """Save the fitted vectorizer and LDA model locally."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(lda, LDA_PATH)
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"LDA model saved to: {LDA_PATH}")


def load_topic_model():
    """Load the saved vectorizer and LDA model."""
    if not VECTORIZER_PATH.exists() or not LDA_PATH.exists():
        raise FileNotFoundError(
            "LDA model files not found. Train the topic model first."
        )
    return joblib.load(VECTORIZER_PATH), joblib.load(LDA_PATH)
