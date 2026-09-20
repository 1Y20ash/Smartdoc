"""Train and evaluate the SmartDoc LDA-based document classifier."""

from src.classifier import build_classifier, evaluate_classifier, save_classifier
from src.dataset import load_dataset
from src.preprocessing import preprocess_documents
from src.topic_model import get_document_topic_distribution, load_topic_model


def main():
    print("Loading prepared dataset...")
    train_df, test_df = load_dataset()

    print(f"Training documents: {len(train_df)}")
    print(f"Testing documents: {len(test_df)}")

    print("Loading existing LDA model...")
    vectorizer, lda = load_topic_model()

    print("Preprocessing training documents...")
    train_documents = preprocess_documents(train_df["text"].tolist())
    print("Generating training topic distributions...")
    train_topics = get_document_topic_distribution(train_documents, vectorizer, lda)

    print("Training Logistic Regression classifier...")
    classifier = build_classifier(train_topics, train_df["category"].tolist())
    save_classifier(classifier)

    print("\nPreprocessing test documents...")
    test_documents = preprocess_documents(test_df["text"].tolist())
    print("Generating test topic distributions...")
    test_topics = get_document_topic_distribution(test_documents, vectorizer, lda)

    print("Evaluating classifier...")
    results = evaluate_classifier(classifier, test_topics, test_df["category"].tolist())

    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results["report"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])
    print("\nClassifier training completed successfully.")


if __name__ == "__main__":
    main()
