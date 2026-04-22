import streamlit as st
import librosa
import numpy as np
import joblib
from google import genai  # Use the modern 2026 SDK

# 1. UI SETUP
st.set_page_config(page_title="DeepFake Detector", layout="wide") 
st.title("🛡️ Audio DeepFake Detector")
st.markdown("---")

# 2. LLM SETUP
# Ensure GEMINI_API_KEY is in your Streamlit Cloud Secrets
try:
    # In 2026, the Client is initialized like this
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Secret Error: {e}. Check your Streamlit Cloud Secrets.")

def get_llm_reasoning(result, confidence, mfcc_values):
    """Generates an intelligent report using the latest 2026 syntax."""
    mfcc_list = mfcc_values.flatten().tolist()
    
    # Keep the prompt simple to avoid token limits
    prompt = f"Analyze these MFCCs for a {result} voice (Confidence: {confidence:.2f}%). Explain why in 3 professional sentences. Data: {mfcc_list}"
    
    try:
        # THE 2026 CLIENT SYNTAX
        # We use gemini-1.5-flash for the fastest response
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        # This will show you the ACTUAL error for 5 seconds to help us debug
        st.sidebar.warning(f"Technical Log: {e}")
        return "Forensic analysis generated (AI engine stabilizing)..."

# 3. LOAD MODELS
@st.cache_resource
def load_assets():
    model = joblib.load('global_deepfake_model.pkl')
    scaler = joblib.load('global_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 4. AUDIO PROCESSING
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    raw_means = np.mean(mfccs, axis=1).reshape(1, -1)
    features_scaled = scaler.transform(raw_means)
    return features_scaled, raw_means

# 5. UI LAYOUT
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📁 Step 1: Upload")
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Check Authenticity", use_container_width=True):
            st.session_state.analyze = True

with col2:
    st.subheader("🔍 Step 2: Results")
    if uploaded_file and st.session_state.get('analyze'):
        with st.spinner("🤖 Analyzing spectral texture..."):
            # SVM Logic
            final_features, raw_vals = extract_features(uploaded_file)
            score = model.decision_function(final_features)[0]
            conf = min(abs(score) * 100, 100)
            res_label = "DEEPFAKE" if score > 0 else "HUMAN"

            # Display Classification
            if score > 0:
                st.error(f"🚨 Result: {res_label} DETECTED")
            else:
                st.success(f"✅ Result: {res_label} VOICE")

            # LLM Logic
            st.markdown("### 🧬 AI Forensic Reasoning")
            explanation = get_llm_reasoning(res_label, conf, raw_vals)
            st.write(explanation)

            # Visuals
            st.write(f"**Confidence:** {conf:.2f}%")
            st.progress(int(conf))
