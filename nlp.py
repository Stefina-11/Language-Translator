import streamlit as st
from transformers import pipeline
from langdetect import detect
st.title("NLP Language Detection & Translation")
input_text = st.text_area("Enter your text here:")
if input_text:
    try:
        detected_lang = detect(input_text)
        st.write(f"Detected language: {detected_lang}")
        model_name = f"Helsinki-NLP/opus-mt-{detected_lang}-en"
        translator = pipeline("translation", model=model_name)
        translation = translator(input_text, max_length=400)
        translated_text = translation[0]['translation_text']
        st.write("Translated text:")
        st.text_area("", translated_text, height=150)
    except Exception as e:
        st.error(f"Error: {e}")
