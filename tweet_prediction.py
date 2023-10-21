import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the pre-trained CountVectorizer and model
loaded_vectorizer = joblib.load('saved/count_vectorizer.pkl')
loaded_model = joblib.load('saved/lr_model.pkl')

# Streamlit app
st.title("Tweet Sentiment Analysis")

# Text input for the tweet
tweet = st.text_area("Enter your tweet:")

if st.button("Predict"):
    if tweet:
        # Transform the input text using the loaded CountVectorizer
        X_new = loaded_vectorizer.transform([tweet])

        # Make a prediction using the loaded model
        prediction = loaded_model.predict(X_new)[0]

        # Display the prediction result
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Please enter a tweet for prediction.")
