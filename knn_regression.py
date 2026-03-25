import streamlit as st
import joblib
import numpy as np

st.title("KNN Regression Prediction App")

# Load model safely
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Input fields
credit_score = st.number_input("Credit Score")
income = st.number_input("Income")
loan_amount = st.number_input("Loan Amount")

# Prediction
if st.button("Predict"):
    prediction = model.predict(np.array([[credit_score, income, loan_amount]]))
    st.success(f"Prediction: {prediction[0]}")