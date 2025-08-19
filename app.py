import streamlit as st
import joblib
import os

st.title("Amazon Alexa Review Sentiment Analysis")

BASE_DIR = os.path.dirname(__file__)
vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")
model_path = os.path.join(BASE_DIR, "sentiment_model.joblib")

try:
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model or vectorizer files are missing!")
    st.stop()

review = st.text_input("Enter your review:")

if review:
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)
    st.write("Sentiment:", "Positive" if prediction[0] == 1 else "Negative")
