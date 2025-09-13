# Language-Translator
An intelligent Python-based NLP tool that detects the language of input text and translates it into English using Hugging Face Transformers and Helsinki-NLP models. Supports dynamic multilingual translation with high accuracy and minimal configuration.

📌 Features

🌍 Automatic Language Detection using langdetect

🔁 Dynamic Translation Model Selection (e.g., opus-mt-fr-en, opus-mt-de-en, etc.)

🌐 Accurate Translation to English using Hugging Face’s transformers library

⚡ Real-Time Translation Output for a wide range of languages

🛠️ Technologies Used
Python

Hugging Face Transformers – for accessing powerful multilingual translation models

Langdetect – for lightweight language identification

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
    (Note: A `requirements.txt` file will be created if it doesn't exist.)
3.  **Run the NLP script:**
    ```bash
    python nlp.py
    ```
    Follow the prompts to enter text for language detection and translation.

## 📋 Requirements

*   Python 3.7+
*   `transformers` library
*   `langdetect` library
*   `torch` or `tensorflow` (depending on your backend preference for Hugging Face Transformers)
