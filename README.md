# Language-Translator
An intelligent Python-based NLP tool that detects the language of input text and translates it into a selected target language using Hugging Face Transformers and the NLLB-200 model. Supports dynamic multilingual translation with high accuracy and minimal configuration.

📌 Features

🌍 Automatic Language Detection using fastText

🔁 Dynamic Translation Model Selection (NLLB-200 supports over 200 languages)

🌐 Accurate Translation to Selected Language using Hugging Face’s transformers library

⚡ Real-Time Translation Output for a wide range of languages

🛠️ Technologies Used
Python

Hugging Face Transformers – for accessing powerful multilingual translation models

fastText – for robust language identification

## 🚀 How to Use

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Stefina-11/Language-Translator.git
    cd Language-Translator
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Note: You will also need to download the `lid.176.bin` model from the fastText official site: `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin` and place it in the project directory.)
3.  **Run the NLP script:**
    ```bash
    streamlit run nlp.py
    ```
    This will open the application in your web browser. Follow the prompts to enter text for language detection and translation.

## 📋 Requirements

*   Python 3.7+
*   `streamlit` library
*   `transformers` library
*   `fasttext` library
*   `torch` or `tensorflow` (depending on your backend preference for Hugging Face Transformers)
