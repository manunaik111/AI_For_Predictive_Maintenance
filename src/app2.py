import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy.stats
import numpy.fft as fft
import os

# --- Configuration ---
MODEL_DIR = 'C:\project file 002\models'
SCALERS_DIR = 'C:\project file 002\scalers001' # Make sure this matches where train_femto_model saved it

# Load V9 Models
REG_MODEL_PATH = os.path.join(MODEL_DIR, 'regressor_v9_femto.pkl')
CLASS_MODEL_PATH = os.path.join(MODEL_DIR, 'classifier_v9_femto.pkl')
SCALER_PATH = os.path.join(SCALERS_DIR, 'scaler_X_v9_femto.pkl')

WINDOW_SIZE = 2560
SAMPLING_RATE = 25600
# Failure Threshold in V9 was "files remaining" (50 files). 
# Let's convert that to a user-friendly "Danger Zone" unit.
FAILURE_THRESHOLD_FILES = 50 

# --- Feature Extraction Function (V9 - 22 Features) ---
def extract_features_v9(data, window_size, sampling_rate, load_val, temp_val):
    """
    Extracts the 22 features expected by the V9 model.
    Note: Since user upload usually has only 1 vibration column,
    we will simulate the 2nd sensor by duplicating the features 
    or using the same signal for both (Horizontal & Vertical).
    This is a necessary approximation for the demo.
    """
    if len(data) < window_size:
        st.warning(f"Data too short. Padding.")
        padding = np.zeros(window_size - len(data))
        segment = np.concatenate([data, padding])
    else:
        segment = data[-window_size:]

    # Helper to calculate 10 features for one signal
    def get_10_features(sig):
        mean = np.mean(sig)
        std = np.std(sig)
        rms = np.sqrt(np.mean(sig**2))
        kurt = scipy.stats.kurtosis(sig)
        skew = scipy.stats.skew(sig)
        peak = np.max(np.abs(sig))
        crest = peak / rms if rms != 0 else 0
        
        freqs = np.fft.fftfreq(len(sig), d=1.0/sampling_rate)[:len(sig)//2]
        fft_mag = np.abs(np.fft.fft(sig))[:len(sig)//2]
        if len(freqs) > 0:
            fft_peak = freqs[np.argmax(fft_mag)]
        else:
            fft_peak = 0
        fft_mean = np.mean(fft_mag)
        fft_std = np.std(fft_mag)
        return [mean, std, rms, kurt, skew, peak, crest, fft_peak, fft_mean, fft_std]

    # 1. Calculate features for "Horizontal" sensor (Real Data)
    feats_h = get_10_features(segment)
    
    # 2. Calculate features for "Vertical" sensor (Simulated/Duplicate)
    # In a real deployment, we'd ask for a 2nd file. For this demo, we reuse the signal.
    feats_v = get_10_features(segment) 

    # 3. Combine: [10 H-Vib] + [10 V-Vib] + [1 Temp] + [1 Load]
    all_features = np.concatenate([feats_h, feats_v, [temp_val], [load_val]])
    
    return all_features.reshape(1, -1)

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        with open(REG_MODEL_PATH, 'rb') as f: reg = pickle.load(f)
        with open(CLASS_MODEL_PATH, 'rb') as f: clf = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
        return reg, clf, scaler
    except Exception as e:
        return None, None, None

model_reg, model_class, scaler_X = load_models()

if model_reg is None:
    st.error(f"Models not found. Looked for: {REG_MODEL_PATH}")
    st.stop()

# --- Interface ---
st.set_page_config(layout="wide", page_title="Predictive Maintenance AI")
st.title("⚙ AI for Predictive Maintenance")
st.subheader("Advanced Multi-Sensor Analysis ")
st.markdown("---")

# --- Sidebar inputs ---
st.sidebar.header("Sensor Fusion Inputs")
st.sidebar.info("Configure the additional sensor parameters manually if not present in the uploaded file.")

manual_load = st.sidebar.number_input("Operational Load (Newtons)", 0, 10000, 5000, 100, help="Load force in Newtons (N)")
manual_temp = st.sidebar.number_input("Bearing Temperature (°C)", 0, 200, 50, 1, help="Current operating temperature")

uploaded_file = st.file_uploader("Upload Vibration Data (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Handle generic CSVs
        if 'Vibration' in df.columns:
            bearing_data = df['Vibration'].values
        else:
            # Fallback: assume first column is vibration
            bearing_data = df.iloc[:, 0].values

        # Display Inputs
        st.sidebar.success("File Processed")
        st.subheader("Input Signal Analysis")
        st.line_chart(bearing_data[:500])
        
        # --- Run Prediction Pipeline ---
        # 1. Extract Features (using manual Load & Temp)
        features = extract_features_v9(bearing_data, WINDOW_SIZE, SAMPLING_RATE, manual_load, manual_temp)
        
        # 2. Scale
        features_scaled = scaler_X.transform(features)
        
        # 3. Predict
        pred_files_remaining = model_reg.predict(features_scaled)[0]
        pred_class = model_class.predict(features_scaled)[0]
        pred_proba = model_class.predict_proba(features_scaled)[0]
        
        # Convert predictions to readable time (assuming 10s per file)
        estimated_seconds_left = pred_files_remaining * 90
        
        # --- Display Results ---
        st.markdown("---")
        st.header("AI Diagnosis Results")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.metric("Est. Remaining Life", f"{estimated_seconds_left:.1f} hours")
            st.caption(f"Raw Prediction: {pred_files_remaining:.1f} test intervals remaining")
            
        with c2:
             # Temperature Check
             if manual_temp > 70:
                 st.metric("Temp Status", f"{manual_temp}°C (HIGH)", delta="-Overheating", delta_color="inverse")
             else:
                 st.metric("Temp Status", f"{manual_temp}°C (Normal)", delta="Optimal")

        with c3:
            if pred_class == 1:
                st.metric("Overall Health", "⚠ CRITICAL FAILURE", delta_color="inverse")
                st.error(f"Failure Probability: {pred_proba[1]*100:.1f}%")
                st.write("Recommendation: STOP MACHINE IMMEDIATELY")
            else:
                st.metric("Overall Health", "✅ OPERATIONAL")
                st.success(f"Health Probability: {pred_proba[0]*100:.1f}%")
                st.write("Recommendation: Continue normal operation")

    except Exception as e:
        st.error(f"Analysis Error: {e}")

else:
    st.info("Waiting for vibration data...")