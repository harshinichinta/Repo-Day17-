import streamlit as st
import joblib
import numpy as np

st.title("Loan Prediction App")

# Load model
model = joblib.load("model.pkl")

# User inputs
credit_score = st.number_input("Credit Score")
income = st.number_input("Income")
loan_amount = st.number_input("Loan Amount")

# Prediction
if st.button("Predict"):
    prediction = model.predict(np.array([[credit_score, income, loan_amount]]))
    st.success(f"Prediction: {prediction[0]}")