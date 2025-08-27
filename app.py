# app.py
import streamlit as st
import time
import tempfile
import subprocess
from transformers import pipeline
import imageio_ffmpeg as iio_ffmpeg

# ---------------- Streamlit Page Config ---------------- #
st.set_page_config(page_title="🎤 Speech-to-Text Demo - By Rayyan Ahmed", layout="centered")

# ---------------- Background ---------------- #
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
    url("https://dopetgztsfho3.cloudfront.net/Open_AI_Whisper_8d5d1dd5e5.webp");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}
h1 {
    color: #FFD700;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ---------------- #
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: rgba(0,50,70,0.6); color: white; }
[data-testid="stSidebar"] .css-1v3fvcr { color: white; }
[data-testid="stSidebar"] h1, h2, h3 { color: #00171F; }
::-webkit-scrollbar-thumb { background: #00cfff; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar.expander("📁 Project Intro"):
    st.markdown(
        "This demo app converts speech to text using OpenAI's Whisper model. "
        "Upload audio in WAV, MP3, or M4A format and get an instant transcription."
    )
    st.markdown("- **⚙️ Model:** openai/whisper-tiny")
    st.markdown("- **Framework:** 🤗 Transformers + Streamlit")
    st.markdown("- ⚡ Lightweight model, suitable for small demos")

with st.sidebar.expander("👨‍💻 Developer's Intro"):
    st.markdown("- **Hi, I'm Rayyan Ahmed**")
    st.markdown("- **Google Certified AI Prompt Specialist**")
    st.markdown("- **IBM Certified Advanced LLM FineTuner**")
    st.markdown("- **Google Certified Soft Skill Professional**")
    st.markdown("- **Hugging Face Certified in Fundamentals of LLMs**")
    st.markdown("- **Expertise in EDA, ML, ANN, CNN, CV, RNN, NLP, LLMs**")
    st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/rayyan-ahmed-504725321/)")

with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown("- Python")
    st.markdown("- Streamlit")
    st.markdown("- Transformers")
    st.markdown("- OpenAI Whisper Model")
    st.markdown("- imageio-ffmpeg (for audio conversion)")
    st.markdown("- Tempfile, Time (built-in)")

# ---------------- Load ASR Model ---------------- #
@st.cache_resource
def load_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

asr = load_model()

# ---------------- Helper: Convert audio to WAV ---------------- #
def convert_to_wav(uploaded_file, target_rate=16000):
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Output WAV temp file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav.close()

    # Use imageio-ffmpeg binary
    ffmpeg_exe = iio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-i", tmp_path,
        "-ac", "1",
        "-ar", str(target_rate),
        "-vn",
        "-y",
        temp_wav.name
    ]
    subprocess.run(cmd, check=True)

    return temp_wav.name

# ---------------- Streamlit UI ---------------- #
st.title("🎤 Speech-to-Text with Whisper")
st.markdown('<h4 style="color:white;">Developed by Rayyan Ahmed</h4>', unsafe_allow_html=True)
st.write("Upload an audio file and get transcription instantly.")

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🚀 Transcribe"):
        with st.spinner("⏳ Transcribing... please wait..."):
            start = time.time()

            # Convert audio to WAV and run ASR
            wav_file = convert_to_wav(uploaded_file)
            result = asr(wav_file, return_timestamps=True)

            end = time.time()

        st.success(f"✅ Transcription complete! (Time taken: {end - start:.2f}s)")
        st.text_area("📝 Transcribed Text:", result["text"], height=200)
