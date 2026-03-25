import streamlit as st
import librosa
import numpy as np
import joblib

# 1. UI SETUP
st.set_page_config(page_title="DeepFake Detector")
st.title("Audio DeepFake Detector")


# 2. LOAD BOTH FILES
@st.cache_resource
def load_assets():
    # Loading both files you just saved in Kaggle
    model = joblib.load('global_deepfake_model.pkl')
    scaler = joblib.load('global_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 3. FEATURE EXTRACTION
# 3. FEATURE EXTRACTION (Bulletproof Version)
def extract_features(audio_path):
    # 1. Force the sampling rate to 22050 (The standard for your trained model)
    y, sr = librosa.load(audio_path, sr=22050, res_type='kaiser_fast')
    
    # 2. Extract exactly 40 MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # 3. Average across time (Resulting in a 1D array of 40 numbers)
    mfccs_processed = np.mean(mfccs, axis=1)
    
    # 4. Reshape to (1, 40) so the Scaler sees "1 Sample with 40 Traits"
    features_reshaped = mfccs_processed.reshape(1, -1)
    
    # 5. Transform using the Scaler from Kaggle
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
