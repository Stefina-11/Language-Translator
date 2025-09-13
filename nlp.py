import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import fasttext
import os

@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

@st.cache_resource
def load_fasttext_model():
    if not os.path.exists("lid.176.bin"):
        st.error("Download 'lid.176.bin' from fastText official site: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
        st.stop()
    return fasttext.load_model("lid.176.bin")

ft_model = load_fasttext_model()

lang_map = {
    "English": "eng_Latn",
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Chinese (Simplified)": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Russian": "rus_Cyrl",
    "Arabic": "arb_Arab",
    "Hindi": "hin_Deva"
}

st.title("NLP Language Detection & Translation")
input_text = st.text_area("Enter your text:")

if input_text:
    detected = ft_model.predict(input_text.replace("\n", " "))  # clean newlines
    detected_lang_code = detected[0][0].replace("__label__", "")
    confidence = detected[1][0]

    human_readable_detected_lang = detected_lang_code
    for key, value in lang_map.items():
        if value.startswith(detected_lang_code):
            human_readable_detected_lang = key
            break

    st.write(f"**Detected language:** {human_readable_detected_lang} (confidence: {confidence:.2f})")

    initial_target_lang_key = None
    for key, value in lang_map.items():
        if value.startswith(detected_lang_code):
            initial_target_lang_key = key
            break
    
    initial_index = 0
    if initial_target_lang_key:
        initial_index = list(lang_map.keys()).index(initial_target_lang_key)

    search_query = st.text_input("Search target language:", "")

    filtered_languages = [
        lang for lang in lang_map.keys() if search_query.lower() in lang.lower()
    ]

    if initial_target_lang_key and initial_target_lang_key in filtered_languages:
        initial_index = filtered_languages.index(initial_target_lang_key)
    else:
        initial_index = 0 # Reset if the detected language is not in the filtered list

    selected_target_lang = st.selectbox("Select target language:", filtered_languages, index=initial_index)
    target_lang = lang_map[selected_target_lang]

    if target_lang.startswith(detected_lang_code):
        st.info("Source and target languages are the same. No translation needed.")
        st.text_area("Output:", input_text, height=150)
    else:
        inputs = tokenizer(input_text, return_tensors="pt")
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
            max_length=400
        )
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        st.success("Translation complete!")
        st.text_area("Translated text:", translated_text, height=150)
else:
    search_query = st.text_input("Search target language:", "")

    filtered_languages = [
        lang for lang in lang_map.keys() if search_query.lower() in lang.lower()
    ]
    selected_target_lang = st.selectbox("Select target language:", filtered_languages)
    target_lang = lang_map[selected_target_lang]
