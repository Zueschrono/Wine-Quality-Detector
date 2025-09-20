import streamlit as st
import numpy as np
import joblib


model = joblib.load("model.joblib")
imputer = joblib.load("imputer.joblib")
scaler = joblib.load("scaler.joblib")


features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]


st.set_page_config(
    page_title="Wine Quality Predictor 🍷",
    page_icon="🍷",
    layout="centered"
)


st.title("🍇Wine Quality Predictor")
st.markdown(
    """ 
    Use this tool to input the chemical properties of a wine sample and discover whether it’s likely to be **premium quality**.  
    Powered by a trained machine learning model. 🧪🍷
    """
)


st.header("🔬 Enter Wine Details:")

user_input = []
cols = st.columns(2)

for i, feat in enumerate(features):
    with cols[i % 2]:
        val = st.number_input(f"{feat.title()}", min_value=0.0, format="%.1f")
        user_input.append(val)


if st.button("🍷 Predict Wine Quality"):
    # Prepare input
    X_new = np.array(user_input).reshape(1, -1)
    X_new_imputed = imputer.transform(X_new)
    X_new_scaled = scaler.transform(X_new_imputed)


    pred = model.predict(X_new_scaled)[0]
    proba = model.predict_proba(X_new_scaled)[0, 1]


    st.subheader("📊 Prediction Result")
    if pred == 1:
        st.success("🎉 **Good Quality Wine!**")
        st.balloons()
    else:
        st.warning("🍷 **Bad Quality Wine!**")

    st.metric(label="Confidence Score", value=f"{proba:.1%}")
    st.caption("The confidence score indicates how likely the wine is to be of good quality.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'><small>Crafted with ❤️ by Rencent Claud. 2025</small></div>",
    unsafe_allow_html=True
)