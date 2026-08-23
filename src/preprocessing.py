"""Text preprocessing utilities for SmartDoc."""


def clean_text(text: str) -> str:
    """Return cleaned document text.

    The detailed NLP preprocessing pipeline will be added after the
    training dataset is selected.
    """
    return " ".join(str(text).split())
