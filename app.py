import streamlit as st
import librosa
import numpy as np
import joblib
from google import genai  # <--- The New Import

# 1. UI SETUP
st.set_page_config(page_title="DeepFake Detector", layout="wide") 
st.title("🛡️ Audio DeepFake Detector")

# 2. NEW LLM SETUP (google-genai style)
try:
    # The new SDK automatically looks for 'GEMINI_API_KEY' in environment variables
    # or you can pass it explicitly like this:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("Gemini API Key not found in Streamlit Secrets.")

def get_llm_reasoning(result, confidence, mfcc_values):
    mfcc_list = mfcc_values.flatten().tolist()
    
    prompt = f"""
    You are a forensic audio expert. 
    Result: {result}
    Confidence: {confidence:.2f}%
    Data: {mfcc_list}

    Write a 3-sentence report explaining the classification. Focus on spectral 
    textures and robotic artifacts vs human resonance.
    """
    
    try:
        # The new syntax uses client.models.generate_content
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return "Forensic reasoning is being processed..."

# ... (rest of your code remains the same)


# 3. LOAD ASSETS
@st.cache_resource
def load_assets():
    model = joblib.load('global_deepfake_model.pkl')
    scaler = joblib.load('global_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 4. FEATURE EXTRACTION
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    raw_means = np.mean(mfccs, axis=1).reshape(1, -1)
    features_scaled = scaler.transform(raw_means)
    return features_scaled, raw_means

# 5. TWO-COLUMN LAYOUT
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader(" Step 1: Upload Audio")
    uploaded_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])
    
    if uploaded_file:
        st.info("Audio file ready for analysis.")
        st.audio(uploaded_file)
        analyze_btn = st.button("Check Authenticity", use_container_width=True)

with col2:
    st.subheader(" Step 2: Analysis Results")
    
    if uploaded_file and analyze_btn:
        with st.spinner(" AI is performing forensic analysis..."):
            # A. Process Audio
            final_features, raw_vals = extract_features(uploaded_file)
            
            # B. SVM Logic
            decision_score = model.decision_function(final_features)[0]
            confidence = min(abs(decision_score) * 100, 100) 
            result_text = "DEEPFAKE" if decision_score > 0 else "HUMAN"

            # C. Display Binary Result
            if decision_score > 0:
                st.error(f" Result: {result_text} DETECTED")
            else:
                st.success(f" Result: {result_text} VOICE")

            # D. The LLM Reasoning (The "Why")
            st.markdown("###  AI Forensic Reasoning")
            reasoning = get_llm_reasoning(result_text, confidence, raw_vals)
            st.write(reasoning)

            # E. Confidence Bar
            st.write(f"**Model Confidence:** {confidence:.2f}%")
            st.progress(int(confidence))

            # F. System Metadata
            with st.expander(" System Metadata"):
                st.json({
                    "Model": "Non-linear SVM-RBF",
                    "Input Dimension": "13-MFCC Vector Fingerprint",
                    "Normalization": "Z-Score Scaled",
                    "Inference Engine": "Scikit-Learn 1.6.1"
                })
    else:
        st.write("Perform analysis to see the LLM-generated forensic report.")
