"""Supervised document classification using LDA topic distributions."""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
CLASSIFIER_PATH = MODEL_DIR / "classifier.joblib"


def build_classifier(topic_distributions, labels, random_state: int = 42):
    """Train Logistic Regression on document-topic distributions."""
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )
    classifier.fit(topic_distributions, labels)
    return classifier


def evaluate_classifier(classifier, topic_distributions, labels):
    """Return standard classification metrics for the classifier."""
    predictions = classifier.predict(topic_distributions)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "report": classification_report(labels, predictions),
        "confusion_matrix": confusion_matrix(labels, predictions),
        "predictions": predictions,
    }


def save_classifier(classifier):
    """Save the trained classifier locally."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, CLASSIFIER_PATH)
    print(f"Classifier saved to: {CLASSIFIER_PATH}")


def load_classifier():
    """Load the saved Logistic Regression classifier."""
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            "Classifier model not found. Train the classifier first."
        )
    return joblib.load(CLASSIFIER_PATH)
