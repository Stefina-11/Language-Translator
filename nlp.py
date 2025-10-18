import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()


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

def get_target_language():
    target_lang_options = list(lang_map.keys())
    selected_lang = st.selectbox("Select target language:", target_lang_options, key="target_language_select")
    return lang_map[selected_lang] if selected_lang else None

target_language = get_target_language()
if target_language:
    pass
else:
    st.warning("Please select a target language.")

if st.button("Translate"):
    if target_language and input_text:
        # Language detection using transformers pipeline
        # Using a smaller, dedicated language identification model
        lang_id_pipeline = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
        lang_id_result = lang_id_pipeline(input_text)[0]
        detected_lang_code = lang_id_result['label'].split('_')[0].lower() # e.g., 'en' from 'en_XX'
        confidence = lang_id_result['score']

        human_readable_detected_lang = detected_lang_code
        for key, value in lang_map.items():
            if value.startswith(detected_lang_code):
                human_readable_detected_lang = key
                break

        st.write(f"**Detected language:** {human_readable_detected_lang} (confidence: {confidence:.2f})")

        target_language_name = next((key for key, value in lang_map.items() if value == target_language), None)
        if target_language_name and human_readable_detected_lang == target_language_name:

            st.info("The detected language is the same as the target language.")

        inputs = tokenizer(input_text, return_tensors="pt")
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_language),
            max_length=400
        )
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        st.success("Translation complete!")
        st.text_area("Translated text:", translated_text, height=150)
    else:
        st.warning("Please select a language and enter text.")
