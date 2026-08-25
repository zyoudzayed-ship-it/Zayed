import os
import subprocess
from flask import Flask, request, send_file, render_template_string
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

app = Flask(__name__)

# تحميل نموذج ديموكس مرة واحدة
model = get_model('htdemucs')

HTML_PAGE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="utf-8">
    <title>أداة عزل الموسيقى</title>
    <style>
        body { font-family: Tahoma, sans-serif; background: #f4f4f9; text-align: center; padding: 50px; }
        .box { background: white; padding: 30px; border-radius: 10px; display: inline-block; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
        input, button { margin-top: 15px; padding: 10px; font-size: 16px; }
        button { background: #ff4757; color: white; border: none; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="box">
        <h2>🎤 أداة عزل الموسيقى عن الصوت والفيديو</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="video/*,audio/*" required><br>
            <button type="submit">بدء الفصل الآن</button>
        </form>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file:
            input_path = "temp_input"
            uploaded_file.save(input_path)
            
            clean_wav = "clean.wav"
            # تحويل أي فيديو أو صوت مباشرة عبر ffmpeg النظامي المثبت عبر apt.txt
            subprocess.run(f"ffmpeg -y -i {input_path} -ac 2 -ar 44100 {clean_wav}", shell=True, check=True)
            
            wav, sr = torchaudio.load(clean_wav)
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()

            sources = apply_model(model, wav[None], shifts=0, split=True, overlap=0.25)[0]
            sources = sources * ref.std() + ref.mean()

            vocals_path = "vocals.wav"
            torchaudio.save(vocals_path, sources[3], sr)
            
            return send_file(vocals_path, as_attachment=True)
            
    return render_template_string(HTML_PAGE)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

