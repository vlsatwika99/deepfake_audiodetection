import streamlit as st
import librosa
import numpy as np
import joblib

# 1. UI SETUP
st.set_page_config(page_title="DeepFake Detector", layout="wide") # Use wide mode for side-by-side
st.title("🛡️ Audio DeepFake Detector")
st.markdown("---")

# 2. LOAD ASSETS
@st.cache_resource
def load_assets():
    model = joblib.load('global_deepfake_model.pkl')
    scaler = joblib.load('global_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 3. FEATURE EXTRACTION
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_processed = np.mean(mfccs, axis=1).reshape(1, -1)
    features_scaled = scaler.transform(mfccs_processed)
    return features_scaled 

# 4. TWO-COLUMN LAYOUT (SIDE-BY-SIDE CARDS)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📁 Step 1: Upload Audio")
    uploaded_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])
    
    if uploaded_file:
        st.info("Audio Loaded Successfully")
        st.audio(uploaded_file)
        analyze_btn = st.button("Check Authenticity", use_container_width=True)

with col2:
    st.subheader("🔍 Step 2: Analysis Results")
    
    if uploaded_file and analyze_btn:
        with st.spinner("Analyzing spectral patterns..."):
            # Process
            final_features = extract_features(uploaded_file)
            
            # Get Distance from Hyperplane
            decision_score = model.decision_function(final_features)[0]
            confidence = min(abs(decision_score) * 100, 100) 

            # Results Display
            if decision_score > 0:
                st.error("🚨 Result: DEEPFAKE DETECTED")
                reason = "The model detected high-frequency mathematical artifacts and 'robotic' spectral buzzing typical of AI voice cloning."
            else:
                st.success("✅ Result: HUMAN VOICE")
                reason = "The acoustic resonance and 13-dimensional MFCC fingerprint align with natural human biological speech patterns."

            # Confidence Bar
            st.write(f"**Model Confidence:** {confidence:.2f}%")
            st.progress(int(confidence))

          
    else:
        st.write("Results will appear here after analysis.")

