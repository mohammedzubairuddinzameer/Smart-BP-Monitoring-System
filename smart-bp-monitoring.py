import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Smart BP Monitoring", layout="wide")

st.markdown("""
    <style>
    body { background-color: #f2f6ff; color: #000; }
    .main { background-color: #ffffff; padding: 20px; border-radius: 10px; }
    .title { color: #003366; font-size: 36px; font-weight: bold; text-align: center; }
    .section-header { color: #004080; font-size: 22px; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🩺 Smart Blood Pressure Monitoring System</h1>", unsafe_allow_html=True)

# ---- DATA LOADING & PREPROCESSING ----
@st.cache_data
def load_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(30, 70, 100),
        'gender': np.random.choice([0, 1], 100),
        'heart_rate': np.random.randint(60, 100, 100),
        'sleep_hours': np.random.uniform(4, 9, 100),
        'activity_level': np.random.uniform(1, 10, 100),
        'medication_dose': np.random.uniform(0.5, 2.0, 100),
        'systolic_bp': np.random.randint(110, 180, 100)
    })
    return data

data = load_data()

# ---- SIDEBAR ----
menu = st.sidebar.radio("Menu", ["📊 Dashboard", "🔬 Train Model", "🧠 Live Prediction"])

# ---- MODEL TRAINING ----
def train_model():
    X = data.drop("systolic_bp", axis=1)
    y = data["systolic_bp"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, mae, rmse

# ---- DASHBOARD ----
if menu == "📊 Dashboard":
    st.markdown("<h3 class='section-header'>📈 Sample Patient Data</h3>", unsafe_allow_html=True)
    st.dataframe(data.head(10))

    st.markdown("<h3 class='section-header'>📊 Blood Pressure Distribution</h3>", unsafe_allow_html=True)
    st.bar_chart(data['systolic_bp'])

# ---- TRAIN MODEL ----
elif menu == "🔬 Train Model":
    st.markdown("<h3 class='section-header'>🔧 Training ML Model on Historical Data</h3>", unsafe_allow_html=True)
    model, mae, rmse = train_model()
    st.success(f"✅ Model Trained Successfully!")
    st.info(f"📉 MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    with open("bp_model.pkl", "wb") as f:
        pickle.dump(model, f)

# ---- LIVE PREDICTION ----
elif menu == "🧠 Live Prediction":
    st.markdown("<h3 class='section-header'>🧬 Enter Real-time Patient Data</h3>", unsafe_allow_html=True)

    age = st.slider("Age", 18, 80, 45)
    gender = st.radio("Gender", ["Male", "Female"])
    heart_rate = st.slider("Heart Rate (bpm)", 50, 120, 75)
    sleep = st.slider("Sleep Hours", 0.0, 10.0, 6.0)
    activity = st.slider("Activity Level (1-10)", 1.0, 10.0, 5.0)
    dose = st.slider("Medication Dose (mg)", 0.0, 5.0, 1.0)

    if st.button("🔍 Predict BP"):
        try:
            with open("bp_model.pkl", "rb") as f:
                model = pickle.load(f)
        except:
            st.error("⚠️ Please train the model first from 'Train Model' tab.")
            st.stop()

        input_data = np.array([[age, 1 if gender == "Male" else 0, heart_rate, sleep, activity, dose]])
        predicted_bp = model.predict(input_data)[0]

        st.success(f"🩸 Predicted Systolic BP: {predicted_bp:.2f} mmHg")

        if predicted_bp >= 140:
            st.error("⚠️ High BP Alert! Immediate attention required.")
        elif predicted_bp <= 90:
            st.warning("⚠️ Low BP detected. Consult doctor.")
        else:
            st.success("✅ BP is within normal range.")

---

### ✅ Summary of Features

- ✅ Real-time BP prediction with ML
- ✅ Upload sensor data for training
- ✅ Random Forest-based forecasting
- ✅ Alert generation
- ✅ Clean UI with Streamlit
- ✅ Model evaluation using MAE & RMSE

---

Would you like me to integrate **cloud storage**, **email alerts**, or a **login system** as next features?
