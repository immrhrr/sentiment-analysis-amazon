
import re
import joblib
import streamlit as st
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load artifacts
vectorizer = joblib.load('artifacts/tfidf_vectorizer.joblib')
model = joblib.load('artifacts/sentiment_model.joblib')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in stop_words]
    return " ".join(tokens)

st.title("Amazon Alexa Review Sentiment")
review = st.text_area("Enter a review:")
if st.button("Predict"):
    cleaned = preprocess(review)
    X = vectorizer.transform([cleaned])
    prob = model.predict_proba(X)[0,1]
    pred = int(prob >= 0.5)
    st.write({"prediction": pred, "probability": float(prob)})
