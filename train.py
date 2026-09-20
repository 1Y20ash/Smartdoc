"""Train and inspect the SmartDoc LDA topic model."""

from src.dataset import load_dataset
from src.preprocessing import preprocess_documents
from src.topic_model import (
    build_topic_model,
    get_top_words_per_topic,
    save_topic_model,
)


N_TOPICS = 8


def main():
    print("Loading prepared dataset...")
    train_df, _ = load_dataset()

    print(f"Training documents: {len(train_df)}")
    print("Preprocessing documents...")
    documents = preprocess_documents(train_df["text"].tolist())

    print("Training LDA topic model...")
    vectorizer, lda = build_topic_model(
        documents,
        n_topics=N_TOPICS,
        max_features=10000,
        min_df=5,
        max_df=0.95,
        random_state=42,
    )

    save_topic_model(vectorizer, lda)

    print("\nDiscovered topics:")
    topics = get_top_words_per_topic(vectorizer, lda, n_words=10)
    for topic_number, words in topics.items():
        print(f"Topic {topic_number}: {', '.join(words)}")

    print("\nLDA training completed successfully.")


if __name__ == "__main__":
    main()
