"""Reusable NLP preprocessing utilities for SmartDoc."""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


_STOPWORDS = None
_LEMMATIZER = None


def download_nltk_resources() -> None:
    """Download the NLTK resources required by SmartDoc if missing."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name, quiet=False)


def _get_stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        download_nltk_resources()
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def _get_lemmatizer():
    global _LEMMATIZER
    if _LEMMATIZER is None:
        download_nltk_resources()
        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


def clean_text(text: str) -> str:
    """Clean and lemmatize English document text.

    Steps:
    1. Convert to lowercase.
    2. Remove URLs, punctuation, and numbers.
    3. Tokenize on whitespace.
    4. Remove English stopwords and very short tokens.
    5. Lemmatize remaining words.
    """
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()

    stops = _get_stopwords()
    lemmatizer = _get_lemmatizer()

    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stops and len(token) > 2
    ]

    return " ".join(cleaned_tokens)


def preprocess_documents(documents):
    """Clean a sequence of documents using the same pipeline."""
    return [clean_text(document) for document in documents]
