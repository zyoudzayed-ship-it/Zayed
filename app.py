import os

os.system("pip install gradio demucs torch torchaudio")

import gradio as gr
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

model = get_model('htdemucs')

def separate_audio(audio_path):
    if audio_path is None:
        return None, None
    
    wav, sr = torchaudio.load(audio_path)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    
    sources = apply_model(model, wav[None], shifts=0, split=True, overlap=0.25)[0]
    sources = sources * ref.std() + ref.mean()
    
    vocals_path = "vocals.wav"
    music_path = "music.wav"
    
    torchaudio.save(vocals_path, sources[3], sr)
    torchaudio.save(music_path, sources[0] + sources[1] + sources[2], sr)
    
    return vocals_path, music_path

with gr.Blocks(title="أداة عزل الموسيقى") as demo:
    gr.Markdown("# 🎙️ أداة عزل الموسيقى عن الصوت")
    gr.Markdown("رفع الملف لعزل الموسيقى منه فوراً وبجودة عالية.")
    
    audio_input = gr.Audio(type="filepath", label="ارفع ملف الصوت أو الفيديو هنا")
    btn = gr.Button("بدء الفصل الان", variant="primary")
    
    with gr.Column():
        vocal_output = gr.Audio(label="🎤 الصوت البشري (بدون موسيقى)")
        music_output = gr.Audio(label="🎸 الموسيقى وحدها")
    
    btn.click(
        fn=separate_audio,
        inputs=[audio_input],
        outputs=[vocal_output, music_output]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)

