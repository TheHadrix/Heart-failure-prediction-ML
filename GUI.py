import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Configuration
st.set_page_config(page_title="Heart Disease Screening", layout="wide")
st.title("🫀 Clinical Heart Disease Prediction")
st.markdown("---")

# Load Model and Scaler
@st.cache_resource
def load_assets():
    model = joblib.load('medical_lr_model.pkl')
    scaler = joblib.load('medical_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.subheader("Patient Clinical Data:")

# UI Form
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex", ["M", "F"])
    resting_bp = st.number_input("Resting BP", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 (FastingBS)", [0, 1])
    max_hr = st.number_input("Maximum Heart Rate (MaxHR)", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=-2.0, max_value=6.0, value=1.0)

with col3:
    chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
    resting_ecg = st.selectbox("Resting ECG", ["LVH", "Normal", "ST"])
    st_slope = st.selectbox("ST Slope", ["Down", "Flat", "Up"])

st.markdown("---")

if st.button("🔍 Run Clinical Analysis", use_container_width=True):
    
    # 1. Map Binary Variables
    sex_val = 1 if sex == "M" else 0
    exang_val = 1 if exercise_angina == "Y" else 0
    
    # 2. Scale Numeric Variables
    numeric_features = pd.DataFrame(
        [[age, resting_bp, cholesterol, max_hr, oldpeak]], 
        columns=['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    )
    scaled_numeric = scaler.transform(numeric_features)[0]
    
    # 3. Manual One-Hot Encoding (drop_first=True equivalent)
    # ChestPainType (Assuming ASY is dropped)
    cp_ata = 1 if chest_pain == "ATA" else 0
    cp_nap = 1 if chest_pain == "NAP" else 0
    cp_ta  = 1 if chest_pain == "TA" else 0
    
    # RestingECG (Assuming LVH is dropped)
    ecg_normal = 1 if resting_ecg == "Normal" else 0
    ecg_st     = 1 if resting_ecg == "ST" else 0
    
    # ST_Slope (Assuming Down is dropped)
    slope_flat = 1 if st_slope == "Flat" else 0
    slope_up   = 1 if st_slope == "Up" else 0
    
    # 4. Assemble the exact 15 features in the correct order
    input_data = np.array([[
        scaled_numeric[0], # Age
        sex_val,           # Sex
        scaled_numeric[1], # RestingBP
        scaled_numeric[2], # Cholesterol
        fasting_bs,        # FastingBS
        scaled_numeric[3], # MaxHR
        exang_val,         # ExerciseAngina
        scaled_numeric[4], # Oldpeak
        cp_ata, cp_nap, cp_ta,       # ChestPain dummies
        ecg_normal, ecg_st,          # RestingECG dummies
        slope_flat, slope_up         # ST_Slope dummies
    ]])
    
    # 5. Prediction
    disease_probability = model.predict_proba(input_data)[0, 1]
    threshold = 0.3
    
    st.subheader("Analysis Result:")
    
    if disease_probability >= threshold:
        st.error("⚠️ Warning: High Risk of Heart Disease Detected")
        st.warning(f"Calculated Probability: {disease_probability:.1%} (Threshold: 30%)")
    else:
        st.success("✅ Patient Status: Low Risk")
        st.info(f"Calculated Probability: {disease_probability:.1%} (Threshold: 30%)")