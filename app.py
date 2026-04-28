import streamlit as st
import librosa
import numpy as np
import joblib
import os
from google import genai  # Modern 2026 SDK


st.set_page_config(page_title="DeepFake Detector", layout="wide") 
st.title(" Audio DeepFake Detector")
st.markdown("---")


try:
    
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Secret Error: {e}. Check your Streamlit Cloud Secrets.")

def get_llm_reasoning(result, confidence, raw_values):
    """Generates an intelligent forensic report based on 39 spectral features."""
   
    data_snippet = raw_values.flatten()[:10].tolist()
    
    prompt = f"""
    Context: Forensic Audio Analysis for Deepfake Detection.
    Detected Class: {result}
    Model Confidence: {confidence:.2f}%
    Partial Spectral Fingerprint: {data_snippet}
    
    Task: Write a 3-sentence professional explanation. 
    Explain how spectral contrast (texture) and harmonic consistency (Chroma) 
    helped distinguish this {result} voice from a synthetic/real counterpart.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        st.sidebar.warning(f"AI Technical Log: {e}")
        return "The AI reasoning engine is currently busy. The SVM/RF classification is complete, but the text summary is pending."


@st.cache_resource
def load_assets():
    # These files must match your LFS push from earlier
    model = joblib.load('global_deepfake_model.pkl')
    scaler = joblib.load('global_scaler.pkl')
    return model, scaler

model, scaler = load_assets()


def extract_features(audio_path):
   
    y, sr = librosa.load(audio_path, sr=16000)
    
    
    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)
    
  
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
    

    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    
 
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    

    combined = np.hstack([mfccs, contrast, chroma]).reshape(1, -1)
    

    features_scaled = scaler.transform(combined)
    return features_scaled, combined


col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader(" Step 1: Upload Audio")
    uploaded_file = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3"])
    
    if uploaded_file:
        st.info("Audio loaded. Ready for analysis.")
        st.audio(uploaded_file)
        if st.button("Check Authenticity", use_container_width=True):
            st.session_state.analyze = True

with col2:
    st.subheader(" Step 2: Analysis Results")
    
    if uploaded_file and st.session_state.get('analyze'):
        with st.spinner(" Analyzing harmonics and spectral texture..."):
        
            final_features, raw_vals = extract_features(uploaded_file)
            
       
            probs = model.predict_proba(final_features)[0]
            
          
         
            if probs[1] > 0.5:
                res_label = "DEEPFAKE"
                conf = probs[1] * 100
                st.error(f" Result: {res_label} DETECTED")
            else:
                res_label = "HUMAN"
                conf = probs[0] * 100
                st.success(f" Result: {res_label} VOICE")

           
            st.markdown("###  AI Forensic Reasoning")
            explanation = get_llm_reasoning(res_label, conf, raw_vals)
            st.write(explanation)

         
            st.write(f"**Verification Confidence:** {conf:.2f}%")
            st.progress(int(conf))

          
            with st.expander(" System Metadata"):
                st.json({
                    "Model": "Random Forest (Ensemble)",
                    "Features": "39 (MFCC, Contrast, Chroma)",
                    "Sample Rate": "16,000 Hz",
                    "Normalization": "Enabled"
                })
    else:
        st.write("Awaiting audio input to begin forensic deep-scan...")
