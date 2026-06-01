import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objects as go

# ----------------------------
# Load Models
# ----------------------------
model_rnn = load_model("simple_rnn_sentiment_model.h5")
model_lstm = load_model("lstm_sentiment_model.h5")
model_gru = load_model("gru_sentiment_model.h5")

# ----------------------------
# Load Tokenizer
# ----------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LENGTH = 200

# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        seq,
        maxlen=MAX_LENGTH,
        padding='post',
        truncating='post'
    )
    return padded

# ----------------------------
# Prediction Function
# ----------------------------
def predict_sentiment(model, review):

    processed = preprocess_text(review)

    prediction = model.predict(processed, verbose=0)[0][0]

    sentiment = "Positive" if prediction > 0.5 else "Negative"

    confidence = prediction if prediction > 0.5 else (1 - prediction)

    positive_prob = prediction
    negative_prob = 1 - prediction

    return sentiment, confidence, positive_prob, negative_prob

# ----------------------------
# UI
# ----------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    layout="wide"
)

st.title("🎬 Movie Review Sentiment Analysis System")

st.subheader(
    "Deep Learning Based Sentiment Classification"
)

st.markdown("---")

# ----------------------------
# Model Selection
# ----------------------------

selected_model = st.radio(
    "Select Model",
    ["SimpleRNN", "LSTM", "GRU"]
)

review = st.text_area(
    "Enter your movie review here...",
    height=150
)

# ----------------------------
# Analyze Button
# ----------------------------

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        if selected_model == "SimpleRNN":
            model = model_rnn

        elif selected_model == "LSTM":
            model = model_lstm

        else:
            model = model_gru

        sentiment, confidence, pos_prob, neg_prob = predict_sentiment(
            model,
            review
        )

        st.success(f"Sentiment: {sentiment}")

        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )

        # ------------------------
        # Probability Chart
        # ------------------------

        st.subheader("Probability Distribution")

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=["Positive", "Negative"],
                y=[pos_prob, neg_prob]
            )
        )

        fig.update_layout(
            title="Positive vs Negative Probability",
            yaxis_title="Probability"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ------------------------
        # Confidence Gauge
        # ------------------------

        st.subheader("Confidence Chart")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={"text": "Confidence %"},
                gauge={
                    "axis": {"range": [0, 100]}
                }
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

# ----------------------------
# Compare All Models
# ----------------------------

st.markdown("---")

st.header("Compare All Models")

if st.button("Compare Predictions"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        results = []

        for name, model in [
            ("SimpleRNN", model_rnn),
            ("LSTM", model_lstm),
            ("GRU", model_gru)
        ]:

            sentiment, confidence, _, _ = predict_sentiment(
                model,
                review
            )

            results.append({
                "Model": name,
                "Sentiment": sentiment,
                "Confidence (%)": round(
                    confidence * 100,
                    2
                )
            })

        st.dataframe(results)

        comparison_fig = go.Figure()

        comparison_fig.add_trace(
            go.Bar(
                x=[r["Model"] for r in results],
                y=[r["Confidence (%)"] for r in results]
            )
        )

        comparison_fig.update_layout(
            title="Model Confidence Comparison",
            yaxis_title="Confidence (%)"
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True
        )