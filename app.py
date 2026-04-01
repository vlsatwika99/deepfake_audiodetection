import streamlit as st
import librosa
import numpy as np
import joblib

# 1. UI SETUP
st.set_page_config(page_title="DeepFake Detector")
st.title("🛡️ Audio DeepFake Detector")

# 2. LOAD ASSETS
@st.cache_resource
def load_assets():
    # Ensure these .pkl files are in your GitHub repo
    model = joblib.load('global_deepfake_model.pkl')
    scaler = joblib.load('global_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 3. FEATURE EXTRACTION
def extract_features(audio_path):
    # Standardize sampling rate to 22.05kHz
    y, sr = librosa.load(audio_path, sr=22050, res_type='kaiser_fast')
    
    # Extract 13 MFCCs to match your Global Model
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Calculate Mean and Reshape to (1, 13)
    mfccs_processed = np.mean(mfccs, axis=1).reshape(1, -1)
    
    # Transform using the Scaler
    features_scaled = scaler.transform(mfccs_processed)
    
    return features_scaled 

# 4. APP INTERFACE
uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Check Authenticity"):
        with st.spinner("Analyzing spectral fingerprints..."):
            # --- EVERYTHING BELOW MUST BE INDENTED ---
            
            # 1. Extract AND Scale
            final_features = extract_features(uploaded_file)
            
            # 2. Get the "Distance" from the boundary (Hyperplane)
            # Positive = Deepfake, Negative = Human
            decision_score = model.decision_function(final_features)[0]
            confidence = min(abs(decision_score) * 100, 100) 

            # 3. Show Result based on the Score
            if decision_score > 0:
                st.error("🚨 Result: DEEPFAKE DETECTED")
                reason = "The model detected high-frequency mathematical artifacts typical of AI cloning."
            else:
                st.success("✅ Result: HUMAN VOICE")
                reason = "The vocal tract resonance patterns align with natural human speech characteristics."

            # 4. Attractive UI Elements
            st.write(f"**Confidence Level:** {confidence:.2f}%")
            st.progress(int(confidence))

            with st.expander("Why this result?"):
                st.write(f"**Technical Insight:** {reason}")
                st.info("This decision is based on a 13-dimensional MFCC fingerprint mapped against a non-linear SVM-RBF boundary.")
