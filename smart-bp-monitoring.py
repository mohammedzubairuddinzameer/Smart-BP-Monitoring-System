import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# --- Page Config ---
st.set_page_config(page_title="Personalized Blood Pressure Prediction", layout="centered")

# --- CSS for Modern UI ---
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        padding: 3rem;
        border-radius: 10px;
    }
    .stTextInput>div>div>input {
        background-color: #f9f9f9;
        padding: 10px;
        font-size: 16px;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        margin-top: 20px;
    }
    h1, h2, h3 {
        color: #1e1e1e;
    }
    </style>
""", unsafe_allow_html=True)

# --- Mock Model Training (you can replace with actual trained model) ---
def train_dummy_model():
    np.random.seed(42)
    data = {
        "age": np.random.randint(20, 80, 300),
        "weight": np.random.randint(40, 100, 300),
        "height": np.random.randint(150, 190, 300),
        "bmi": np.random.uniform(18.0, 35.0, 300),
        "heart_rate": np.random.randint(60, 100, 300)
    }
    df = pd.DataFrame(data)
    df["systolic"] = 100 + 0.5 * df["age"] + 0.2 * df["weight"] - 0.1 * df["height"] + 0.3 * df["heart_rate"]
    df["diastolic"] = 60 + 0.3 * df["age"] + 0.1 * df["weight"] - 0.2 * df["height"] + 0.4 * df["heart_rate"]

    model_sys = RandomForestRegressor()
    model_dia = RandomForestRegressor()

    X = df[["age", "weight", "height", "bmi", "heart_rate"]]
    y_sys = df["systolic"]
    y_dia = df["diastolic"]

    model_sys.fit(X, y_sys)
    model_dia.fit(X, y_dia)
    
    return model_sys, model_dia

model_sys, model_dia = train_dummy_model()

# --- App UI ---
st.title("🩺 Personalized Blood Pressure Prediction")
st.markdown("### Input")

with st.form("bp_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        height = st.number_input("Height (cm)", min_value=100, max_value=220, value=170)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=160, value=72)
    with col2:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        bmi = st.number_input("BMI", min_value=10.0, max_value=40.0, value=22.0)

    submitted = st.form_submit_button("Predict")

# --- Prediction Logic ---
if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "weight": weight,
        "height": height,
        "bmi": bmi,
        "heart_rate": heart_rate
    }])

    systolic = round(model_sys.predict(input_data)[0], 1)
    diastolic = round(model_dia.predict(input_data)[0], 1)

    st.markdown("### Prediction")
    st.markdown(f"**Systolic Blood Pressure:** `{systolic} mmHg`")
    st.markdown(f"**Diastolic Blood Pressure:** `{diastolic} mmHg`")
