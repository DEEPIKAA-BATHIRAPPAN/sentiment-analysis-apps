"""Prediction interface for the trained sentiment model."""

from pathlib import Path
import pickle
from typing import Any

from sentiment_package.preprocessing import preprocess_text


class SentimentAnalyzer:
    """Load the bundled classifier and provide single and batch predictions."""

    MODEL_DIR = Path(__file__).resolve().parent / "models"
    MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
    VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

    def __init__(
        self,
        model_path: str | Path | None = None,
        vectorizer_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else self.MODEL_PATH
        self.vectorizer_path = (
            Path(vectorizer_path) if vectorizer_path else self.VECTORIZER_PATH
        )
        missing = [
            str(path)
            for path in (self.model_path, self.vectorizer_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Trained model files are missing: "
                + ", ".join(missing)
                + ". Run `python scripts/train_model.py` from the project root."
            )

        with self.model_path.open("rb") as model_file:
            self.model: Any = pickle.load(model_file)
        with self.vectorizer_path.open("rb") as vectorizer_file:
            self.vectorizer: Any = pickle.load(vectorizer_file)

    @staticmethod
    def _validate_text(text: str) -> str:
        cleaned = preprocess_text(text)
        if not cleaned:
            raise ValueError("Text must not be empty or whitespace only.")
        return cleaned

    def predict_with_confidence(self, text: str) -> dict[str, Any]:
        """Return the predicted label, its probability, and probabilities for all classes."""
        cleaned = self._validate_text(text)
        features = self.vectorizer.transform([cleaned])
        probabilities = self.model.predict_proba(features)[0]
        classes = self.model.classes_
        probability_by_class = {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities)
        }
        sentiment = str(classes[probabilities.argmax()])
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": probability_by_class[sentiment],
            "probabilities": probability_by_class,
        }

    def predict(self, text: str) -> str:
        """Return only the predicted sentiment label."""
        return self.predict_with_confidence(text)["sentiment"]

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict sentiment for each text, preserving input order."""
        if isinstance(texts, (str, bytes)) or not isinstance(texts, list):
            raise TypeError("Batch input must be a list of strings.")
        return [self.predict_with_confidence(text) for text in texts]
