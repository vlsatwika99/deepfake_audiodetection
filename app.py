import streamlit as st
import librosa
import numpy as np
import joblib
from google import genai  # Latest 2026 SDK

# 1. UI SETUP
st.set_page_config(page_title="DeepFake Detector", layout="wide") 
st.title("🛡️ Audio DeepFake Detector")
st.markdown("---")

# 2. LLM SETUP (Using the modern Client-based approach)
# This looks for your key in Streamlit Cloud Settings > Secrets
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("Missing Gemini API Key. Please add 'GEMINI_API_KEY' to Streamlit Secrets.")

def get_llm_reasoning(result, confidence, mfcc_values):
    """Generates an intelligent forensic report using the 13-MFCC data."""
    # Convert numpy data to a plain list for the prompt
    mfcc_list = mfcc_values.flatten().tolist()
    
    prompt = f"""
    You are a forensic audio expert. A machine learning model analyzed a voice.
    Result: {result}
    Model Confidence: {confidence:.2f}%
    Data Fingerprint: {mfcc_list}

    Task: Write a 3-sentence professional report. 
    Explain why this was classified as {result} based on 'spectral artifacts' versus 'human resonance'.
    Do not use technical jargon; make it understandable for a courtroom.
    """
    
    try:
        # Modern 2026 Client-side generation call
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        # If the LLM fails, we return this message
        return "The SVM model has finished, but the AI forensic summary is still generating..."

# 3. LOAD ASSETS (Kaggle models)
@st.cache_resource
def load_assets():
    model = joblib.load('global_deepfake_model.pkl')
    scaler = joblib.load('global_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 4. FEATURE EXTRACTION
def extract_features(audio_path):
    # Standardize to 22.05kHz
    y, sr = librosa.load(audio_path, sr=22050, res_type='kaiser_fast')
    # 13 MFCCs to match your Global Scaler
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    raw_means = np.mean(mfccs, axis=1).reshape(1, -1)
    # Z-score normalization
    features_scaled = scaler.transform(raw_means)
    return features_scaled, raw_means

# 5. FRONTEND LAYOUT
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📁 Step 1: Upload Audio")
    uploaded_file = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3"])
    
    if uploaded_file:
        st.info("Audio processed. Click button to analyze.")
        st.audio(uploaded_file)
        analyze_btn = st.button("Check Authenticity", use_container_width=True)

with col2:
    st.subheader("🔍 Step 2: Analysis Results")
    
    if uploaded_file and analyze_btn:
        with st.spinner("🤖 AI is performing forensic analysis..."):
            # A. Process the file
            final_features, raw_vals = extract_features(uploaded_file)
            
            # B. SVM Boundary Calculation
            decision_score = model.decision_function(final_features)[0]
            confidence = min(abs(decision_score) * 100, 100) 
            result_label = "DEEPFAKE" if decision_score > 0 else "HUMAN"

            # C. Show Result
            if decision_score > 0:
                st.error(f"🚨 Result: {result_label} DETECTED")
            else:
                st.success(f"✅ Result: {result_label} VOICE")

            # D. The LLM Report
            st.markdown("### 🧬 AI Forensic Reasoning")
            report = get_llm_reasoning(result_label, confidence, raw_vals)
            st.write(report)

            # E. Confidence Gauge
            st.write(f"**Model Confidence Level:** {confidence:.2f}%")
            st.progress(int(confidence))

            # F. Technical Specs
            with st.expander("⚙️ System Metadata"):
                st.json({
                    "Engine": "SVM-RBF + Gemini 1.5",
                    "Input": "13-MFCC Fingerprint",
                    "Sampling": "22,050 Hz",
                    "Decision Score": float(decision_score)
                })
    else:
        st.write("Awaiting audio input for classification...")
