import torch
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import requests

MODEL_NAME = "williamhackspeare/LogicLoom-v1"

@st.cache_resource
def load_model():
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

def generate_headlines(articles, model, tokenizer, max_length=128):
    inputs = tokenizer(articles, max_length=max_length, truncation=True, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=5
        )
    return [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]

st.title("📰 Headline Generator")

# User input for text or file upload
input_mode = st.radio("Choose input mode:", ("Enter text", "Upload file"))

articles = []

if input_mode == "Enter text":
    input_text = st.text_area("Enter your text below:")
    if input_text:
        articles = [input_text]
elif input_mode == "Upload file":
    uploaded_file = st.file_uploader("Upload a text file", type="txt")
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        articles = [line.strip() for line in content.splitlines() if line.strip()]

if articles:
    st.write(f"Loaded {len(articles)} article(s). Generating headlines...")

    # Load the model and tokenizer
    model, tokenizer = load_model()

    # Generate headlines
    headlines = generate_headlines(articles, model, tokenizer)

    for i, (article, headline) in enumerate(zip(articles, headlines), start=1):
        st.subheader(f"Article {i}")
        st.text_area("Article Text", article, height=150, disabled=True)
        st.text_input("Generated Headline", headline, disabled=True)