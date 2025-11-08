import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

@st.cache_resource
def load_translation_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model
@st.cache_resource
def load_lang_detector():
    return pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

tokenizer, model = load_translation_model()
lang_id_pipeline = load_lang_detector()

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
reverse_lang_map = {v.split('_')[0][:2]: v for v in lang_map.values()}
st.title("AI-Powered Language Detection & Translation")
st.markdown("""
Easily detect and translate text across multiple languages using **NLLB-200** (Meta AI) and **XLM-RoBERTa**.
""")
input_text = st.text_area("Enter your text:", height=150, placeholder="Type or paste text here...")

target_lang_name = st.selectbox("Select target language:", list(lang_map.keys()))
target_lang_code = lang_map[target_lang_name] if target_lang_name else None

if st.button("Translate"):
    if not input_text.strip():
        st.warning("Please enter some text to translate.")
    elif not target_lang_code:
        st.warning("Please select a target language.")
    else:
        try:
            # Step 1: Language Detection
            with st.spinner("🔍 Detecting language..."):
                detection = lang_id_pipeline(input_text, truncation=True, max_length=128)[0]
                detected_label = detection['label'].lower()
                confidence = detection['score']

            # Use the detected_label directly as it should be a 2-letter code (e.g., 'en', 'hi')
            detected_lang_code = reverse_lang_map.get(detected_label, "eng_Latn")
            detected_lang_name = next((name for name, code in lang_map.items() if code == detected_lang_code), detected_label)

            st.success(f"Detected Language: **{detected_lang_name}** (Confidence: {confidence:.2f})")

            # Step 2: Skip translation if same language
            if detected_lang_name == target_lang_name:
                st.info("The detected language is the same as the target language — no translation needed.")
            else:
                with st.spinner("Translating... Please wait..."):
                    tokenizer.src_lang = detected_lang_code
                    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

                    # Check for long text truncation
                    if len(tokenizer.tokenize(input_text)) > 512:
                        st.warning("Text truncated to 512 tokens for better performance.")

                    translated_tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang_code),
                        max_length=400
                    )

                    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

                # Step 3: Display Results Side-by-Side
                st.success("Translation Complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Original Text", input_text, height=200)
                with col2:
                    st.text_area("Translated Text", translated_text, height=200)

        except Exception as e:
            st.error("Something went wrong during translation.")
            st.exception(e)
st.markdown("---")
st.caption("Built with ❤️ by Stefina | Powered by Meta NLLB-200 & XLM-RoBERTa")
