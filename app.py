import streamlit as st
import librosa
import numpy as np
import joblib
import os
from google import genai  # Modern 2026 SDK

# 1. UI SETUP
st.set_page_config(page_title="DeepFake Detector", layout="wide") 
st.title("🛡️ Audio DeepFake Detector")
st.markdown("---")

# 2. LLM SETUP (Gemini 3.1 for stability in 2026)
try:
    # Fixed the typo: Client must have a Capital 'C'
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Secret Error: {e}. Check your Streamlit cloud Secrets.")

def get_llm_reasoning(result, confidence, raw_values):
    """Generates an intelligent report based on 39 spectral features."""
    data_snippet = raw_values.flatten()[:10].tolist()
    
    prompt = f"""
    Context: Forensic Audio Analysis.
    Detected: {result}
    Model Confidence: {confidence:.2f}%
    Partial Spectral Data: {data_snippet}
    
    Task: Write a 3-sentence professional explanation. 
    Focus on how spectral contrast and harmonic consistency (Chroma) 
    differentiated this {result} voice from the alternative.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        st.sidebar.warning(f"Technical Log: {e}")
        return "The AI analysis engine is currently stabilizing. Please check back in a moment."

# 3. LOAD MODELS
@st.cache_resource
def load_assets():
    model = joblib.load('global_deepfake_model.pkl')
    scaler = joblib.load('global_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 4. AUDIO PROCESSING (Must match your Kaggle 98% accuracy logic)
def extract_features(audio_path):
    # Standardize to 16kHz to match training
    y, sr = librosa.load(audio_path, sr=16000)
    
    # 1. Silence Trimming & Normalization
    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)
    
    # 2. Extract the "Triple Fingerprint" (Total 39 features)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    
    combined = np.hstack([mfccs, contrast, chroma]).reshape(1, -1)
    
    # 3. Scale the features
    features_scaled = scaler.transform(combined)
    return features_scaled, combined

# 5. UI LAYOUT
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📁 Step 1: Upload")
    uploaded_file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Check Authenticity", use_container_width=True):
            st.session_state.analyze = True

with col2:
    st.subheader("🔍 Step 2: Analysis Results")
    if uploaded_file and st.session_state.get('analyze'):
        with st.spinner("🤖 Analyzing harmonics and spectral texture..."):
            # A. Extract & Predict
            final_features, raw_vals = extract_features(uploaded_file)
            
            # B. Random Forest Probability Logic
            probs = model.predict_proba(final_features)[0]
            
            # C. Determine Label & Confidence
            if probs[1] > 0.5:
                res_label = "DEEPFAKE"
                conf = probs[1] * 100
                st.error(f"🚨 Result: {res_label} DETECTED")
            else:
                res_label = "HUMAN"
                conf = probs[0] * 100
                st.success(f"✅ Result: {res_label} VOICE")

            # D. AI Forensic Reasoning
            st.markdown("### 🧬 AI Forensic Reasoning")
            explanation = get_llm_reasoning(res_label, conf, raw_vals)
            st.write(explanation)

            # E. Visual Confidence Gauge
            st.write(f"**Verification Confidence:** {conf:.2f}%")
            st.progress(int(conf))
