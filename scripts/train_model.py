"""Train, evaluate, and save the sentiment classifier."""

from pathlib import Path
import pickle
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sentiment_package.preprocessing import preprocess_text  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "sentiment_data.csv"
MODEL_DIR = PROJECT_ROOT / "sentiment_package" / "models"
EXPECTED_LABELS = {"Positive", "Negative", "Neutral"}


def train() -> dict[str, object]:
    """Validate the starter data, train the model, and persist both artifacts."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    required_columns = {"text", "sentiment"}
    if not required_columns.issubset(data.columns):
        raise ValueError("Dataset must contain `text` and `sentiment` columns.")
    data = data.loc[:, ["text", "sentiment"]].dropna()
    data["text"] = data["text"].map(preprocess_text)
    data["sentiment"] = data["sentiment"].astype(str).str.strip().str.title()
    data = data[data["text"] != ""]
    labels = set(data["sentiment"].unique())
    if labels != EXPECTED_LABELS:
        raise ValueError(
            f"Dataset labels must be exactly {sorted(EXPECTED_LABELS)}; "
            f"found {sorted(labels)}."
        )
    counts = data["sentiment"].value_counts()
    if (counts < 2).any():
        raise ValueError("Each sentiment label needs at least two examples.")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data["text"],
        data["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=data["sentiment"],
    )
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_features=10000)
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)
    model = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)
    print(f"Dataset rows: {len(data)}")
    print(f"Training rows: {len(train_texts)} | Test rows: {len(test_texts)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(
        "Classification report:\n"
        + classification_report(
            test_labels, predictions, labels=sorted(EXPECTED_LABELS), zero_division=0
        )
    )
    print("Confusion matrix (labels: " + ", ".join(sorted(EXPECTED_LABELS)) + "):")
    print(confusion_matrix(test_labels, predictions, labels=sorted(EXPECTED_LABELS)))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / "sentiment_model.pkl").open("wb") as model_file:
        pickle.dump(model, model_file)
    with (MODEL_DIR / "tfidf_vectorizer.pkl").open("wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    print(f"Saved model and vectorizer to {MODEL_DIR}")
    return {"accuracy": accuracy, "report": classification_report(
        test_labels, predictions, labels=sorted(EXPECTED_LABELS), zero_division=0,
        output_dict=True
    )}


if __name__ == "__main__":
    train()
