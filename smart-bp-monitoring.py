import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
import hashlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="Smart BP Monitoring", layout="wide")

# --- CSS Styling ---
st.markdown("""
    <style>
        body {
            background-color: #f4f4f9;
            color: #1e1e1e;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
        }
        .stButton > button {
            color: white;
            background-color: #3366cc;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Database Connection ---
def get_connection():
    conn = sqlite3.connect("bp_data.db", check_same_thread=False)
    return conn, conn.cursor()

# --- Save patient data to database ---
def save_to_database(age, gender, systolic, diastolic, heart_rate, sleep, activity, prediction):
    conn, cursor = get_connection()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bp_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            gender TEXT,
            systolic REAL,
            diastolic REAL,
            heart_rate INTEGER,
            sleep REAL,
            activity REAL,
            predicted_bp REAL,
            timestamp TEXT
        )
    ''')
    cursor.execute('''
        INSERT INTO bp_logs (age, gender, systolic, diastolic, heart_rate, sleep, activity, predicted_bp, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (age, gender, systolic, diastolic, heart_rate, sleep, activity, prediction, datetime.now().isoformat()))
    conn.commit()

# --- Load Dataset ---
def load_data():
    if os.path.exists("sample_data.csv"):
        df = pd.read_csv("sample_data.csv")
        return df
    else:
        # generate dummy data if not available
        np.random.seed(42)
        df = pd.DataFrame({
            "age": np.random.randint(25, 70, 100),
            "gender": np.random.choice(["Male", "Female"], 100),
            "systolic": np.random.normal(120, 15, 100),
            "diastolic": np.random.normal(80, 10, 100),
            "heart_rate": np.random.randint(60, 100, 100),
            "sleep": np.random.uniform(4, 9, 100),
            "activity": np.random.uniform(0, 10, 100)
        })
        df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
        df["bp_score"] = 0.6 * df["systolic"] + 0.4 * df["diastolic"]
        df.to_csv("sample_data.csv", index=False)
        return df

# --- Model Training ---
def train_model(df):
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1}) if df["gender"].dtype == 'O' else df["gender"]
    X = df[["age", "gender", "heart_rate", "sleep", "activity"]]
    y = 0.6 * df["systolic"] + 0.4 * df["diastolic"]
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

# --- Predict BP Score ---
def predict_bp(model, input_data):
    prediction = model.predict(pd.DataFrame([input_data]))[0]
    return round(prediction, 2)

# --- Main App ---
def main():
    st.title("🩺 Smart Blood Pressure Monitoring System")

    tab1, tab2, tab3 = st.tabs(["📊 Monitor", "📁 History", "ℹ️ About"])

    with tab1:
        st.header("Real-time BP Monitoring")

        # Input
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        systolic = st.slider("Systolic BP", 90, 180, 120)
        diastolic = st.slider("Diastolic BP", 60, 120, 80)
        heart_rate = st.slider("Heart Rate (bpm)", 50, 150, 75)
        sleep = st.slider("Sleep Duration (hours)", 0.0, 12.0, 6.0)
        activity = st.slider("Activity Level (0-10)", 0.0, 10.0, 5.0)

        if st.button("🔍 Predict BP Score"):
            df = load_data()
            model = train_model(df)

            input_data = {
                "age": age,
                "gender": 0 if gender == "Male" else 1,
                "heart_rate": heart_rate,
                "sleep": sleep,
                "activity": activity
            }

            prediction = predict_bp(model, input_data)
            st.success(f"🧠 Predicted BP Score: **{prediction}**")

            save_to_database(age, gender, systolic, diastolic, heart_rate, sleep, activity, prediction)

    with tab2:
        st.header("📁 Patient History")

        conn, cursor = get_connection()
        cursor.execute("SELECT * FROM bp_logs ORDER BY timestamp DESC")
        logs = cursor.fetchall()
        conn.close()

        if logs:
            df = pd.DataFrame(logs, columns=["ID", "Age", "Gender", "Systolic", "Diastolic", "HeartRate", "Sleep", "Activity", "PredictedBP", "Time"])
            st.dataframe(df)
            st.line_chart(df["PredictedBP"])
        else:
            st.info("No history available.")

    with tab3:
        st.header("ℹ️ About This Project")
        st.markdown("""
        This is a **Smart Blood Pressure Monitoring System** developed using **Streamlit** and **Machine Learning**.
        - Real-time BP prediction with ML
        - Tracks user lifestyle like sleep and activity
        - Visual logs and graphs
        - Data stored in local SQLite database

        _Developed for academic or healthcare innovation demo._
        """)

if __name__ == "__main__":
    main()
