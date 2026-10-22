"""Text preprocessing shared by training and inference."""

import re


def preprocess_text(text: str) -> str:
    """Normalize whitespace while leaving useful word and punctuation signals intact."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")
    return re.sub(r"\s+", " ", text).strip()
