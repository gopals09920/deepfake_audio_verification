import os
import streamlit as st
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from src.model import BankingDeepfakeDetector

# Page Configuration
st.set_page_config(
    page_title="Deepfake Voice & Audio Verification System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
        line-height: 1.2;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/security-checked.png", width=70)
    st.title("Deepfake Audio Verification")
    st.markdown("---")
    st.info("**Architecture:**\nWav2Vec 2.0 + 3-Layer CNN")
    st.success("**Dataset:**\nKaggle Deep Voice Detector")
    st.warning("**Application:**\nBanking Fraud Security")
    st.markdown("---")
    st.write("Term Project Dashboard")

# Main Header
st.markdown('<p class="main-title">🛡️ Real-Time Deepfake Voice & Audio Verification</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Banking Fraud Prevention & Biometric Security System</p>', unsafe_allow_html=True)
st.markdown("---")

# Load Model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BankingDeepfakeDetector(num_classes=2).to(device)
    model_path = "models/banking_deepfake_detector.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    return model, device

model, device = load_model()

# Initialize Session State for results persistence
if 'results' not in st.session_state:
    st.session_state.results = {}

# File Uploader (Without any file limit)
st.markdown("### 📂 Step 1: Upload Audio Files")
uploaded_files = st.file_uploader(
    "Select audio clips (.wav, .mp3, .flac)", 
    type=["wav", "mp3", "flac"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.markdown(f"**Total Files Uploaded:** `{len(uploaded_files)}`")
    st.markdown("---")
    
    for i, uploaded_file in enumerate(uploaded_files):
        file_key = f"file_{i}_{uploaded_file.name}"
        
        with st.expander(f"📁 Audio File {i+1}: {uploaded_file.name}", expanded=(i == 0)):
            temp_file_path = f"temp_audio_{i}.wav"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.audio(temp_file_path, format='audio/wav')
            
            # 1. Sabse pehle Run button
            st.markdown("---")
            if st.button(f"🚀 Run Deepfake & Spectral Analysis #{i+1}", key=f"btn_{i}", use_container_width=True):
                with st.spinner("Processing neural inference and generating feature maps..."):
                    try:
                        sample_rate = 16000
                        duration = 3.0
                        max_length = int(sample_rate * duration)

                        audio, sr = librosa.load(temp_file_path, sr=sample_rate)
                        if len(audio) < max_length:
                            audio_padded = np.pad(audio, (0, max_length - len(audio)))
                        else:
                            audio_padded = audio[:max_length]
                            
                        waveform_tensor = torch.tensor(audio_padded, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
                        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                        spec_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                        with torch.no_grad():
                            outputs = model(waveform_tensor, spec_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            confidence, predicted_class = torch.max(probabilities, 1)

                        is_fake = predicted_class.item() == 1
                        conf_score = confidence.item() * 100
                        real_prob = probabilities[0][0].item() * 100
                        fake_prob = probabilities[0][1].item() * 100

                        # Store results in session state
                        st.session_state.results[file_key] = {
                            "is_fake": is_fake,
                            "conf_score": conf_score,
                            "real_prob": real_prob,
                            "fake_prob": fake_prob,
                            "audio_k": audio,
                            "sr_k": sr,
                            "mel_spec_db": mel_spec_db
                        }

                    except Exception as e:
                        st.error(f"Error during execution: {e}")

            # 2. Audio Properties
            audio_k, sr_k = librosa.load(temp_file_path, sr=16000)
            dur = len(audio_k) / sr_k
            
            st.markdown("#### 📊 Audio Properties")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sample Rate", f"{sr_k} Hz")
            c2.metric("Duration", f"{dur:.2f} s")
            c3.metric("Format", uploaded_file.name.split('.')[-1].upper())

            # Display results if available in session state for this file
            if file_key in st.session_state.results:
                res = st.session_state.results[file_key]
                
                st.markdown("---")
                if not res["is_fake"]:
                    st.success(f"### 🛡️ Result: Genuine Voice Verified ({res['conf_score']:.2f}%)")
                else:
                    st.error(f"### 🚨 Result: Deepfake Audio Detected ({res['conf_score']:.2f}%)")

                st.markdown("#### Probability Distribution")
                st.progress(res["real_prob"] / 100.0, text=f"Real Voice Probability: {res['real_prob']:.2f}%")
                st.progress(res["fake_prob"] / 100.0, text=f"Deepfake Probability: {res['fake_prob']:.2f}%")

                st.markdown("---")
                st.markdown("#### 📉 Visual Analytics: Waveform & Mel-Spectrogram")
                
                col_g1, col_g2 = st.columns(2, gap="medium")
                
                with col_g1:
                    st.markdown("**1. Audio Waveform**")
                    fig_w, ax_w = plt.subplots(figsize=(5, 2.5))
                    librosa.display.waveshow(res["audio_k"], sr=res["sr_k"], ax=ax_w, color='#2563EB')
                    ax_w.axis('off')
                    plt.tight_layout()
                    st.pyplot(fig_w)

                with col_g2:
                    st.markdown("**2. Mel-Spectrogram**")
                    fig_s, ax_s = plt.subplots(figsize=(5, 2.5))
                    img = librosa.display.specshow(res["mel_spec_db"], sr=res["sr_k"], x_axis='time', y_axis='mel', ax=ax_s, cmap='Blues')
                    ax_s.axis('off')
                    plt.tight_layout()
                    st.pyplot(fig_s)