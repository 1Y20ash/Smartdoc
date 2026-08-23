"""Dataset preparation utilities for SmartDoc.

AG News is used as the initial real-world document classification dataset.
The raw dataset is downloaded at runtime/development time and is not stored
in the Git repository.
"""

from pathlib import Path
from urllib.request import urlopen

import pandas as pd


LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

AG_NEWS_TRAIN_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
)
AG_NEWS_TEST_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_PATH = DATA_DIR / "ag_news_train.csv"
TEST_PATH = DATA_DIR / "ag_news_test.csv"


def _download_csv(url: str) -> pd.DataFrame:
    """Download one AG News CSV and return it as a DataFrame."""
    with urlopen(url, timeout=60) as response:
        return pd.read_csv(response, header=None)


def download_ag_news() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and normalize the official AG News CSV files."""
    train_raw = _download_csv(AG_NEWS_TRAIN_URL)
    test_raw = _download_csv(AG_NEWS_TEST_URL)
    return prepare_dataset(train_raw, test_raw)


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


def make_balanced_subset(df: pd.DataFrame, samples_per_class: int, random_state: int = 42) -> pd.DataFrame:
    """Return an equally sized, reproducible subset for each category."""
    parts = []
    for category in LABELS.values():
        category_df = df[df["category"] == category]
        if len(category_df) < samples_per_class:
            raise ValueError(
                f"Not enough documents for {category}: "
                f"found {len(category_df)}, need {samples_per_class}."
            )
        parts.append(category_df.sample(samples_per_class, random_state=random_state))

    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


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
