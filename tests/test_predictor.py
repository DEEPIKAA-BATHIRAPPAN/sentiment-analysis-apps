"""Tests for the reusable sentiment analyzer."""

import pytest

from sentiment_package import SentimentAnalyzer


@pytest.fixture(scope="module")
def analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


def test_model_and_vectorizer_load(analyzer: SentimentAnalyzer) -> None:
    assert analyzer.model is not None
    assert analyzer.vectorizer is not None


def test_single_prediction_has_valid_label_and_confidence(
    analyzer: SentimentAnalyzer,
) -> None:
    result = analyzer.predict_with_confidence("I absolutely love this wonderful product.")
    assert result["text"] == "I absolutely love this wonderful product."
    assert result["sentiment"] in {"Positive", "Negative", "Neutral"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"]) == {"Positive", "Negative", "Neutral"}


def test_predict_returns_label(analyzer: SentimentAnalyzer) -> None:
    assert analyzer.predict("The delivery was late and the item was damaged.") in {
        "Positive",
        "Negative",
        "Neutral",
    }


def test_common_positive_and_negative_phrases(analyzer: SentimentAnalyzer) -> None:
    assert analyzer.predict("This is very good.") == "Positive"
    assert analyzer.predict("This is very bad.") == "Negative"


def test_batch_predictions_preserve_order(analyzer: SentimentAnalyzer) -> None:
    texts = ["A delightful experience.", "The package arrived today."]
    results = analyzer.predict_batch(texts)
    assert len(results) == len(texts)
    assert [result["text"] for result in results] == texts


def test_empty_input_is_rejected(analyzer: SentimentAnalyzer) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        analyzer.predict_with_confidence("   ")


def test_invalid_single_input_is_rejected(analyzer: SentimentAnalyzer) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        analyzer.predict_with_confidence(123)  # type: ignore[arg-type]


def test_invalid_batch_input_is_rejected(analyzer: SentimentAnalyzer) -> None:
    with pytest.raises(TypeError, match="list of strings"):
        analyzer.predict_batch("not a list")  # type: ignore[arg-type]


def test_missing_model_files_have_actionable_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="train_model.py"):
        SentimentAnalyzer(tmp_path / "missing-model.pkl", tmp_path / "missing-vectorizer.pkl")
