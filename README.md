# Language-Translator: An Advanced Multilingual NLP Tool

## Overview
The Language-Translator is an intelligent, Python-based Natural Language Processing (NLP) tool designed for seamless language detection and translation. Leveraging state-of-the-art models from Hugging Face Transformers and fastText, this application provides highly accurate, real-time translation across a vast array of languages with minimal configuration. It is ideal for developers, researchers, and users requiring dynamic multilingual communication capabilities.

## Key Features
-   **Automatic Language Detection:** Utilizes `fastText` for robust and accurate identification of the input text's language.
-   **Dynamic Translation Model Selection:** Employs the NLLB-200 model from Hugging Face, supporting translation for over 200 languages.
-   **High-Accuracy Translation:** Achieves precise translations to the selected target language through the `transformers` library.
-   **Real-Time Output:** Delivers instant translation results, enhancing user experience for a wide range of linguistic needs.

## Technologies & Libraries
-   **Python:** The core programming language for the application.
-   **Hugging Face Transformers:** Provides access to powerful pre-trained multilingual translation models, including NLLB-200.
-   **fastText:** Used for efficient and accurate language identification.
-   **Streamlit:** Powers the interactive web-based user interface.
-   **PyTorch / TensorFlow:** Backend deep learning frameworks for Hugging Face Transformers (user's preference).

## Getting Started

### Prerequisites
Ensure you have Python 3.7+ installed on your system.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Stefina-11/Language-Translator.git
    cd Language-Translator
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download fastText language identification model:**
    Download the `lid.176.bin` model from the official fastText site:
    `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin`
    Place the downloaded `lid.176.bin` file directly into the project's root directory.

### Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run nlp.py
    ```
    This command will launch the application in your default web browser.

2.  **Interact with the application:**
    Follow the on-screen prompts to input text for automatic language detection and select your desired target language for translation.

## Contributing
We welcome contributions to the Language-Translator project! Please feel free to fork the repository, create a new branch, and submit a pull request with your enhancements or bug fixes.

## License
This project is licensed under the [LICENSE](LICENSE) file.
