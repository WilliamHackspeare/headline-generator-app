# 📰 Headline Generator

This repository contains a Streamlit app for generating headlines from text using a MarianMT model hosted on the Hugging Face Transformers hub. The model, `willhsp/headline-generator-opus-mt-en-mul`, has been fine-tuned from the `Helsinki-NLP/opus-mt-en-mul` model on the `valurank/News_headlines` dataset (both available on Hugging Face) for headline generation tasks. Users can input text directly or upload a text file containing articles, and the app will generate corresponding headlines.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://headline-generator-willhsp.streamlit.app/)

---

## 🚀 Features
- **Multiple Input Modes:**
  - Enter article text directly in a text area.
  - Upload a text file.
- **High-Quality Headline Generation:**
  - Utilizes the fine-tuned MarianMT model for accurate and concise headlines.
- **User-Friendly Interface:**
  - Interactive Streamlit UI for ease of use.

---

## 🛠 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/WilliamHackspeare/headline-generator-app.git
cd headline-generator-app
```

### 2. Install Dependencies
Ensure Python 3.11 or later is installed. Then, install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run headline_generator_app.py
```

---

## 💾 Model Information
The MarianMT model used in this app is hosted on Hugging Face under the repository:
[`willhsp/headline-generator-opus-mt-en-mul`](https://huggingface.co/willhsp/headline-generator-opus-mt-en-mul).

---

## 📜 Usage Guide

### Input Options
1. **Enter Text Mode:**
   - Select "Enter text" from the input mode options.
   - Paste or type your article text into the provided text area.
2. **Upload File Mode:**
   - Select "Upload file" from the input mode options.
   - Upload a `.txt` file with one article per line.

### Generating Headlines
- Once input is provided, the app will:
  1. Process the input text.
  2. Generate a headline for each article using the model.
  3. Display the original article and the generated headline.

---

## 🔧 Technical Details
- **Model:** MarianMT fine-tuned for headline generation.
- **Frameworks:**
  - [Streamlit](https://streamlit.io/): For interactive UI.
  - [Hugging Face Transformers](https://huggingface.co/transformers): For pretrained models, dataset, and model inference.
- **Deployment:** Can be run locally or deployed on platforms like Streamlit Cloud or Heroku.

---

## 🛡 License
This project is licensed under the MIT License.

---

## 🧩 Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 🌟 Acknowledgements
- Hugging Face for providing the Transformers library.
- Streamlit for making app development simple and elegant.
