import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Streamlit UI
st.title("Amazon Alexa Sentiment Analysis")
st.write("🔍 Classify Amazon Alexa reviews as **Positive** or **Negative**")

# User input
review = st.text_area("Enter a review:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        # Transform input
        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)[0]

        sentiment = "✅ Positive" if prediction == 1 else "❌ Negative"
        st.success(f"Predicted Sentiment: {sentiment}")
