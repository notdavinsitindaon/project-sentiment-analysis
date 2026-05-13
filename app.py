import streamlit as st
from transformers import pipeline
import re

# --- 1. Page Configuration ---
st.set_page_config(page_title="Project Zeitgeist", page_icon="📈", layout="centered")

# --- 2. Load Model (With Caching for Speed) ---
# REPLACE 'YOUR_HF_USERNAME' with your actual Hugging Face username!
MODEL_NAME = "davinsitinda0n/sentimenanalysis"

@st.cache_resource
def load_model():
    # Load the pipeline from Hugging Face Hub
    return pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME)

st.text("Loading AI Model (This takes a few seconds)...")
sentiment_pipeline = load_model()
st.success("Model Loaded Successfully!")

# --- 3. Text Preprocessing Function ---
# (We must clean the text exactly like we did in Kaggle)
def clean_tweet(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '[USER]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 4. User Interface ---
st.title("📈 Project Zeitgeist")
st.subheader("Short-Form Cultural Trend Forecaster")
st.markdown("""
This AI model was fine-tuned on social media data to classify the underlying sentiment of emerging cultural trends. 
Type a sample tweet or Threads post below to see the prediction.
""")

# Text input area
user_input = st.text_area("Enter a social media post:", placeholder="e.g., Digital minimalism is saving my mental health.")

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            # Clean and predict
            cleaned_text = clean_tweet(user_input)
            prediction = sentiment_pipeline(cleaned_text)[0]
            
            # Extract results
            label = "Positive 🟢" if prediction['label'] == 'LABEL_1' else "Negative 🔴"
            confidence = prediction['score'] * 100
            
            # Display results beautifully
            st.markdown("### Prediction Result:")
            st.metric(label="Sentiment", value=label)
            st.progress(prediction['score'])
            st.write(f"*Confidence:* {confidence:.2f}%")