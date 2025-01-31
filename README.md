# 📰 Headline Generator Model

This Streamlit application generates news headlines from article text using a fine-tuned MarianMT machine translation model. The system combines NLP techniques with an intuitive interface for practical use in content creation workflows.

## 🚀 Key Features

- **Dual Input Modes:**
  - Direct text input through text area
  - Batch processing via text file upload
- **Advanced NLP Model:**
  - Fine-tuned MarianMT architecture
  - Beam search decoding with 5 beams
  - Automatic text truncation/padding
- **Production-Ready Implementation:**
  - Model caching with `@st.cache_resource`
  - GPU acceleration support
  - Clean text normalization

## 🛠️ Installation & Setup

1. Clone repository:
```bash
git clone https://github.com/WilliamHackspeare/headline-generator-app.git
cd headline-generator-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch application:
```bash
streamlit run streamlit_app.py
```

## 🧠 Model Architecture

**Base Model:** `Helsinki-NLP/opus-mt-en-mul`  
**Fine-Tuned Version:** [`willhsp/headline-generator-opus-mt-en-mul`](https://huggingface.co/willhsp/headline-generator-opus-mt-en-mul)

**Training Details:**
- Dataset: `valurank/News_headlines` (Hugging Face)
- Input Format: Article text with bullet points
- Output Format: Concise headline
- Training Hardware: NVIDIA T4 GPU
- Evaluation Metric: ROUGE-L, ROUGE-1, BLEU
- Validation Scores:
-   ROUGE-L: 0.7965
-   ROUGE-1: 0.8057
-   BLEU: 0.7816

**Key Technical Specifications:**
```python
MAX_LENGTH = 128  # Tokens
NUM_BEAMS = 5     # Beam search width
```

## 💻 Application Workflow

1. **Model Loading:**
   - Downloads pretrained weights from Hugging Face Hub
   - Initializes tokenizer with MarianMT settings
   - Caches model in memory for subsequent runs

2. **Text Processing:**
   ```python
   def generate_headlines(articles, model, tokenizer):
       inputs = tokenizer(articles, 
                         max_length=128,
                         truncation=True,
                         padding=True,
                         return_tensors="pt")
       
       outputs = model.generate(
           input_ids=inputs["input_ids"],
           attention_mask=inputs["attention_mask"],
           num_beams=5
       )
       return [tokenizer.decode(output, skip_special_tokens=True) 
               for output in outputs]
   ```

3. **User Interface:**
   - Radio button input selection
   - Dynamic text area/file uploader
   - Progress indicators during generation
   - Side-by-side article/headline display



## 📜 License

Apache-2.0 License

## 🔗 Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Streamlit Component Gallery](https://streamlit.io/gallery)
- [MarianMT Paper](https://arxiv.org/abs/1809.00368)
