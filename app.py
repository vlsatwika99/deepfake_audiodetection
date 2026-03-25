import streamlit as st
import librosa
import numpy as np
import joblib

# 1. UI SETUP
st.set_page_config(page_title="DeepFake Detector")
st.title("🛡️ Audio DeepFake Detector")
st.markdown("Machine Learning Journey: Version 2.0 (with Scaler)")

# 2. LOAD BOTH FILES
@st.cache_resource
def load_assets():
    # Loading both files you just saved in Kaggle
    model = joblib.load('global_deepfake_model.pkl')
    scaler = joblib.load('global_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 3. FEATURE EXTRACTION
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    
    # NEW STEP: Scale the features before predicting!
    # We reshape to (1, -1) because the scaler expects a 2D array
    features_reshaped = mfccs_processed.reshape(1, -1)
    features_scaled = scaler.transform(features_reshaped)
    
    return features_scaled

# 4. APP INTERFACE
uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Check Authenticity"):
        with st.spinner("Analyzing..."):
            # Extract AND Scale
            final_features = extract_features(uploaded_file)
            
            # Predict
            prediction = model.predict(final_features)
            
            if prediction[0] == 1:
                st.error("🚨 Result: DEEPFAKE")
            else:
                st.success("✅ Result: HUMAN")