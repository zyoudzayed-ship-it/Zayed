import os
import streamlit as st
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

st.set_page_config(page_title="أداة عزل الموسيقى", page_icon="🎤")

st.title("🎤 أداة عزل الموسيقى عن الصوت")
st.write("رفع الملف لعزل الموسيقى منه فوراً وبجودة عالية.")

@st.cache_resource
def load_demucs_model():
    return get_model('htdemucs')

model = load_demucs_model()

uploaded_file = st.file_uploader("ارفع ملف الصوت هنا", type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file is not None:
    input_path = "temp_input.wav"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("بدء الفصل الآن", type="primary"):
        with st.spinner("جاري معالجة الصوت وعزل الموسيقى... قد يستغرق هذا بضع ثوانٍ"):
            wav, sr = torchaudio.load(input_path)
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()

            sources = apply_model(model, wav[None], shifts=0, split=True, overlap=0.25)[0]
            sources = sources * ref.std() + ref.mean()

            vocals_path = "vocals.wav"
            music_path = "music.wav"

            torchaudio.save(vocals_path, sources[3], sr)
            torchaudio.save(music_path, sources[0] + sources[1] + sources[2], sr)

        st.success("تم الفصل بنجاح!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎤 الصوت البشري")
            st.audio(vocals_path)
        with col2:
            st.subheader("🎵 الموسيقى وحدها")
            st.audio(music_path)
