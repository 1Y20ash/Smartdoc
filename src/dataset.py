"""Dataset preparation utilities for SmartDoc.

AG News is used as the initial real-world document classification dataset.
The raw dataset is downloaded at runtime/development time and is not stored
in the Git repository.
"""

from pathlib import Path

import pandas as pd


LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_PATH = DATA_DIR / "ag_news_train.csv"
TEST_PATH = DATA_DIR / "ag_news_test.csv"


def prepare_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize AG News data into text/category columns."""
    train = train_df.copy()
    test = test_df.copy()

    train.columns = ["label", "title", "description"]
    test.columns = ["label", "title", "description"]

    for frame in (train, test):
        frame["text"] = (
            frame["title"].fillna("").astype(str)
            + " "
            + frame["description"].fillna("").astype(str)
        ).str.strip()
        frame["category"] = frame["label"].map(LABELS)

    return train[["text", "category"]], test[["text", "category"]]


def save_dataset(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Save prepared datasets locally."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prepared local datasets."""
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Prepared AG News files were not found. Run the dataset preparation step first."
        )
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
