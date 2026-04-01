import streamlit as st
import librosa
import numpy as np
import joblib
import pandas as pd

# 1. UI SETUP
st.set_page_config(page_title="DeepFake Detector", layout="wide")
st.title(" Audio DeepFake Detector")
st.markdown("---")

# 2. LOAD ASSETS (Global Model & Scaler)
@st.cache_resource
def load_assets():
    model = joblib.load('global_deepfake_model.pkl')
    scaler = joblib.load('global_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 3. FEATURE EXTRACTION
def extract_features(audio_path):
    # Standardize sampling rate to 22.05kHz
    y, sr = librosa.load(audio_path, sr=22050, res_type='kaiser_fast')
    
    # Extract 13 MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Calculate Mean across the time axis
    mfccs_processed = np.mean(mfccs, axis=1).reshape(1, -1)
    
    # Transform using the Scaler
    features_scaled = scaler.transform(mfccs_processed)
    
    return features_scaled, mfccs_processed

# 4. APP INTERFACE
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload & Playback")
    uploaded_file = st.file_uploader("Choose a WAV or MP3 file", type=["wav", "mp3"])
    if uploaded_file:
        st.audio(uploaded_file)
        analyze_btn = st.button("Check Authenticity", use_container_width=True)

with col2:
    st.subheader(" Analysis Results")
    if uploaded_file and analyze_btn:
        with st.spinner("Analyzing spectral fingerprints..."):
            # Extract features and raw values for visualization
            final_features, raw_mfccs = extract_features(uploaded_file)
            
            # 1. Get Distance from Hyperplane
            decision_score = model.decision_function(final_features)[0]
            confidence = min(abs(decision_score) * 100, 100) 

            # 2. Show Result
            if decision_score > 0:
                st.error("Result: DEEPFAKE DETECTED")
                reason = "The model detected high-frequency mathematical artifacts typical of AI cloning."
            else:
                st.success("Result: HUMAN VOICE")
                reason = "The vocal tract resonance patterns align with natural human speech characteristics."

            # 3. Confidence Gauge
            st.write(f"**Model Confidence:** {confidence:.2f}%")
            st.progress(int(confidence))

            # 4. NEW: Spectral Fingerprint Visualization
            st.write("### Spectral Fingerprint (MFCCs)")
            mfcc_data = pd.DataFrame({
                'Coefficient': [f"MFCC {i+1}" for i in range(13)],
                'Energy Value': raw_mfccs.flatten()
            })
            st.bar_chart(mfcc_data.set_index('Coefficient'))

