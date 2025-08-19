import streamlit as st
import joblib

# Load the pre-trained vectorizer and model
vectorizer = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("sentiment_model.joblib")

st.title("Amazon Alexa Review Sentiment Analysis")

# Get user input
review = st.text_input("Enter your review:")

if review:
    # Transform input using the already fitted vectorizer
    review_tfidf = vectorizer.transform([review])
    
    # Predict sentiment
    prediction = model.predict(review_tfidf)
    
    # Display result
    st.write("Sentiment:", "Positive" if prediction[0] == 1 else "Negative")
