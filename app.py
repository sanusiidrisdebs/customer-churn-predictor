import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Risk Predictor")
st.write("Adjust the customer metrics below to predict churn probability in real time.")

# Load model
#@st.cache_resource
#def load_model():
    #return joblib.load("churn_model.pkl")#

import os

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "churn_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception:
    st.error("Model file 'churn_model.pkl' not found. Run train.py first!")
    st.stop()

# User Inputs
st.subheader("Customer Profile")
tenure = st.slider("Tenure (Months with company)", min_value=1, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=65.0, step=1.0)
total_charges = st.number_input("Total Charges ($)", min_value=15.0, max_value=10000.0, value=float(tenure * monthly_charges), step=10.0)

# Prediction
input_df = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

if st.button("Predict Churn Risk", type="primary"):
    prob = model.predict_proba(input_df)[0][1]
    churn_risk = prob > 0.5
    
    st.markdown("---")
    st.metric(label="Churn Probability", value=f"{prob * 100:.1f}%")
    
    if churn_risk:
        st.error("⚠️ High Risk: This customer is likely to churn. Recommend proactive retention incentives.")
    else:
        st.success("✅ Low Risk: Customer is likely to stay loyal.")