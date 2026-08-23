"""Reusable NLP preprocessing utilities for SmartDoc."""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


_STOPWORDS = None
_LEMMATIZER = None
_RESOURCES_READY = False


def download_nltk_resources() -> None:
    """Download required NLTK resources once if they are missing."""
    global _RESOURCES_READY
    if _RESOURCES_READY:
        return

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

    _RESOURCES_READY = True


def _initialize_nlp():
    """Initialize reusable NLTK objects once per Python process."""
    global _STOPWORDS, _LEMMATIZER
    download_nltk_resources()

    if _STOPWORDS is None:
        _STOPWORDS = set(stopwords.words("english"))
    if _LEMMATIZER is None:
        _LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Clean and lemmatize English document text."""
    _initialize_nlp()

    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()

    cleaned_tokens = [
        _LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in _STOPWORDS and len(token) > 2
    ]

    return " ".join(cleaned_tokens)


def preprocess_documents(documents):
    """Clean a sequence of documents using the same pipeline."""
    _initialize_nlp()
    return [clean_text(document) for document in documents]
