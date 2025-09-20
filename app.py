import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load pre-trained components
model = joblib.load("model.joblib")
imputer = joblib.load("imputer.joblib")
scaler = joblib.load("scaler.joblib")

# Feature names used during training
features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

# Streamlit page config
st.set_page_config(
    page_title="Wine Quality Predictor 🍷",
    page_icon="🍷",
    layout="centered"
)

# Title and intro
st.title("🍇 Wine Quality Predictor")
st.markdown("Powered by a trained machine learning model. 🧪🍷")

# Input header
st.header("🔬 Enter Wine Details:")

# Collect user inputs
user_input = []
cols = st.columns(2)

for i, feat in enumerate(features):
    with cols[i % 2]:
        val = st.number_input(f"{feat.title()}", min_value=0.0, format="%.2f")
        user_input.append(val)

# Predict button
if st.button("🍷 Predict Wine Quality"):
    # Convert input to DataFrame with correct column names
    X_new_df = pd.DataFrame([user_input], columns=features)

    # Apply imputer and scaler
    X_new_imputed = imputer.transform(X_new_df)
    X_new_scaled = scaler.transform(X_new_imputed)

    # Predict quality
    pred = model.predict(X_new_scaled)[0]
    proba = model.predict_proba(X_new_scaled)[0, 1]

    # Show result
    st.subheader("📊 Prediction Result")
    if pred == 1:
        st.success("🎉 **Good Quality Wine!**")
        st.balloons()
    else:
        st.warning("🍷 **Bad Quality Wine!**")

    # Show confidence score
    st.metric(label="Confidence Score", value=f"{proba:.1%}")
    st.caption("The confidence score indicates how likely the wine is to be of good quality.")