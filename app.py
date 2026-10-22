"""Streamlit interface for the reusable sentiment analysis package."""

import pickle

import pandas as pd
import streamlit as st

from sentiment_package import SentimentAnalyzer

st.set_page_config(page_title="Sentiment Studio", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f4f7fb 0%, #eef5f4 100%); }
    .hero { padding: 1.5rem 1.8rem; border-radius: 18px;
      background: linear-gradient(110deg, #123b52, #167d78); color: white; }
    .hero h1 { color: white; margin-bottom: .25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>💬 Sentiment Studio</h1>'
    "<p>Explore the tone of customer feedback with a reusable machine-learning model.</p></div>",
    unsafe_allow_html=True,
)
st.write("")

try:
    analyzer = SentimentAnalyzer()
except FileNotFoundError as exc:
    st.error(f"Model files are not available. Train the model first: {exc}")
    st.stop()
except (OSError, pickle.UnpicklingError) as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

single_tab, batch_tab, about_tab = st.tabs(
    ["Single review", "Batch analysis", "About this project"]
)

with single_tab:
    st.subheader("Analyze a review")
    review = st.text_area(
        "Enter a sentence or review",
        placeholder="For example: The support team was helpful and solved my issue quickly.",
        height=130,
    )
    if st.button("Analyze sentiment", type="primary"):
        if not review.strip():
            st.warning("Enter some text before analyzing.")
        else:
            result = analyzer.predict_with_confidence(review)
            col_result, col_chart = st.columns([1, 1])
            with col_result:
                sentiment = result["sentiment"]
                icon = {"Positive": "😊", "Negative": "🙁", "Neutral": "😐"}[sentiment]
                st.metric("Predicted sentiment", f"{icon} {sentiment}")
                st.metric("Model confidence", f"{result['confidence']:.1%}")
                st.caption(
                    "Confidence is the model's probability for this prediction, "
                    "not a guarantee of correctness or the model's accuracy."
                )
            with col_chart:
                st.caption("Probability by class")
                probabilities = pd.Series(result["probabilities"], name="Probability")
                st.bar_chart(probabilities, y="Probability", color="#16877f")

with batch_tab:
    st.subheader("Analyze multiple reviews")
    pasted_text = st.text_area(
        "Paste one sentence per line",
        placeholder="The product works perfectly.\nThe product broke after one day.",
        height=150,
    )
    uploaded = st.file_uploader("Or upload a CSV containing a `text` column", type=["csv"])
    batch_texts: list[str] = []
    if uploaded is not None:
        try:
            upload_df = pd.read_csv(uploaded)
            if "text" not in upload_df.columns:
                st.error("The uploaded CSV must contain a `text` column.")
            else:
                batch_texts = upload_df["text"].dropna().astype(str).tolist()
                st.caption(f"Loaded {len(batch_texts)} text rows from CSV.")
        except (UnicodeDecodeError, pd.errors.ParserError, OSError) as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
    elif pasted_text.strip():
        batch_texts = [line.strip() for line in pasted_text.splitlines() if line.strip()]

    if st.button("Analyze batch", type="primary", key="analyze_batch"):
        if not batch_texts:
            st.warning("Provide at least one non-empty sentence or upload a valid CSV.")
        else:
            try:
                results = analyzer.predict_batch(batch_texts)
            except (TypeError, ValueError) as exc:
                st.error(f"Could not analyze the batch: {exc}")
            else:
                result_df = pd.DataFrame(
                    [
                        {
                            "text": item["text"],
                            "sentiment": item["sentiment"],
                            "confidence": item["confidence"],
                            **{
                                f"probability_{label.lower()}": probability
                                for label, probability in item["probabilities"].items()
                            },
                        }
                        for item in results
                    ]
                )
                st.dataframe(result_df, use_container_width=True)
                distribution = result_df["sentiment"].value_counts().reindex(
                    ["Positive", "Negative", "Neutral"], fill_value=0
                )
                st.caption("Sentiment distribution")
                st.bar_chart(distribution, color="#16877f")
                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download prediction results",
                    data=csv_bytes,
                    file_name="sentiment_predictions.csv",
                    mime="text/csv",
                )

with about_tab:
    st.subheader("Project information")
    st.write(
        "This academic starter project trains a TF-IDF and Logistic Regression "
        "classifier and exposes inference through the `SentimentAnalyzer` Python package."
    )
    st.code(
        "Dataset → TF-IDF → Logistic Regression → Reusable Python Package "
        "→ Streamlit Application → Cloud Deployment",
        language="text",
    )
    st.info(
        "The bundled examples are a small starter dataset. Real-world performance "
        "depends on representative, sufficiently large and carefully labelled data. "
        "Streamlit Community Cloud hosts this app; this is not an Azure Machine "
        "Learning deployment."
    )
