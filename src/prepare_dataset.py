"""Download and prepare AG News for the SmartDoc development run."""

from src.dataset import download_ag_news, make_balanced_subset, save_dataset


if __name__ == "__main__":
    train, test = download_ag_news()
    train = make_balanced_subset(train, samples_per_class=5000)
    test = make_balanced_subset(test, samples_per_class=1000)
    save_dataset(train, test)

    print(f"Training documents: {len(train)}")
    print(f"Testing documents: {len(test)}")
    print("Training distribution:")
    print(train["category"].value_counts().sort_index())
