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
        detected = ft_model.predict(input_text.replace("\n", " "))  # clean newlines
        detected_lang_code = detected[0][0].replace("__label__", "")
        confidence = detected[1][0]

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
