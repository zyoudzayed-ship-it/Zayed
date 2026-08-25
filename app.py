import os
import streamlit as st
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from pydub import AudioSegment

st.set_page_config(page_title="أداة عزل الموسيقى", page_icon="🎤")

st.title("🎤 أداة عزل الموسيقى عن الصوت والفيديو")
st.write("رفع ملف صوت أو فيديو لعزل الموسيقى عنه فوراً وبجودة عالية.")

@st.cache_resource
def load_demucs_model():
    return get_model('htdemucs')

model = load_demucs_model()

uploaded_file = st.file_uploader(
    "ارفع ملف الصوت أو الفيديو هنا", 
    type=["mp3", "wav", "m4a", "ogg", "mp4", "mov", "mkv", "avi"]
)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    input_path = f"temp_input.{file_extension}"
    converted_audio_path = "temp_audio.wav"
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("بدء الفصل الآن", type="primary"):
        with st.spinner("جاري تحويل الملف وعزل الموسيقى..."):
            try:
                # تحويل الفيديو أو الصوت إلى WAV مستقل لتفادي خطأ TorchCodec
                audio = AudioSegment.from_file(input_path)
                audio.export(converted_audio_path, format="wav")

                wav, sr = torchaudio.load(converted_audio_path)
                
                if wav.shape[0] > 2:
                    wav = wav[:2, :]
                
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
                    
            except Exception as e:
                st.error(f"حدث خطأ أثناء المعالجة: {e}")
