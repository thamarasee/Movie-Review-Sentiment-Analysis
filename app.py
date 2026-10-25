
import re
from pathlib import Path
import joblib
import streamlit as st

st.set_option("client.toolbarMode", "minimal")

# Clean the movie review
def clean_review(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = text.lower()
    text = " ".join(text.split())
    return text

# Load the saved model and vectorizer
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "sentiment_model.pkl")
vectorizer = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")


st.set_page_config(
    page_title="Movie Review Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Analyzer")

with st.sidebar:
    st.title("🎬 About This Project")
    st.write("Movie Review Sentiment Analysis")
    st.divider()

    st.subheader("🤖 Machine Learning Model")
    st.write("Algorithm: Logistic Regression")
    st.write("Feature Extraction: TF-IDF")

    st.subheader("📊 Model Performance")
    st.write("Test Accuracy: Approximately 88.3%")

    st.divider()
    st.caption("Developed using Python and Streamlit.")

st.write(
    "Discover the sentiment behind your movie reviews "
    "using Machine Learning."
)
st.divider()


st.subheader("📝 Enter Your Movie Review")

def clear_review():
    st.session_state["review_input"] = ""

review = st.text_area(
    "Write or paste your review below:",
    height=180,
    placeholder="Example: This movie was amazing! The story was interesting, and the actors performed brilliantly.",
    key="review_input"
)

st.button("Clear Review", on_click=clear_review)


# Predict sentiment
if st.button("Analyze Sentiment"):
    if review.strip():
        cleaned = clean_review(review)
        review_vector = vectorizer.transform([cleaned])
        prediction = model.predict(review_vector)[0]

        probabilities = model.predict_proba(review_vector)[0]
        confidence = max(probabilities) * 100

        if prediction == 1:
            st.success("😊 Positive Review")
            st.write(
                "The model predicts that this review "
                "expresses a positive sentiment."
            )
            st.metric("Prediction Confidence", f"{confidence:.2f}%")
        else:
            st.error("😞 Negative Review")
            st.write(
                "The model predicts that this review "
                "expresses a negative sentiment."
            )
            st.metric("Prediction Confidence", f"{confidence:.2f}%")
    else:
        st.warning("Please enter a movie review.")


st.divider()

st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🎬 Movie Review Analyzer | Powered by Machine Learning"
    "</div>",
    unsafe_allow_html=True
)
