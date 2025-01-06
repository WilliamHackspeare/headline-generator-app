# 📰 Headline Generator

This repository contains a Streamlit app for generating headlines from text using a MarianMT model hosted on the Hugging Face Transformers hub. The model, `williamhackspeare/LogicLoom-v1`, has been fine-tuned for headline generation tasks. Users can input text directly or upload a text file containing articles, and the app will generate corresponding headlines.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://willhsp-logicloom.streamlit.app/)

---

## 🚀 Features
- **Multiple Input Modes:**
  - Enter article text directly in a text area.
  - Upload a text file containing multiple articles (one per line).
- **High-Quality Headline Generation:**
  - Utilizes the fine-tuned MarianMT model for accurate and concise headlines.
- **User-Friendly Interface:**
  - Interactive Streamlit UI for ease of use.
- **Downloadable Output:**
  - Generated headlines can be reviewed and downloaded as a CSV file.

---

## 🛠 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/WilliamHackspeare/logicloom3-app.git
cd logicloom3-app
```

### 2. Install Dependencies
Ensure Python 3.7 or later is installed. Then, install the required Python packages:
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
[`williamhackspeare/LogicLoom-v1`](https://huggingface.co/williamhackspeare/LogicLoom-v1).

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

### Downloading Results
- After headlines are generated, a CSV download option will appear for saving the results.

---

## 🔧 Technical Details
- **Model:** MarianMT fine-tuned for headline generation.
- **Frameworks:**
  - [Streamlit](https://streamlit.io/): For interactive UI.
  - [Hugging Face Transformers](https://huggingface.co/transformers): For model inference.
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