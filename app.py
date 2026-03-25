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
# 3. FEATURE EXTRACTION (The Corrected Version)
def extract_features(audio_path):
    # 1. Force the Sampling Rate to match your Training Data
    y, sr = librosa.load(audio_path, sr=22050, res_type='kaiser_fast')
    
    # 2. Extract exactly 40 MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # 3. Calculate the Mean across the time axis (axis=1)
    # This results in exactly 40 numbers.
    mfccs_processed = np.mean(mfccs, axis=1)
    
    # 4. Reshape to (1, 40)
    features_reshaped = mfccs_processed.reshape(1, -1)
    
    # 5. DEBUG: This will print the shape in your Streamlit Logs
    # It should say (1, 40). If it doesn't, we found the bug!
    print(f"Feature Shape: {features_reshaped.shape}")
    
    # 6. Transform using the Scaler
    features_scaled = scaler.transform(features_reshaped)
    
    return features_scaled features_scaled

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
