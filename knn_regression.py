import streamlit as st
import pandas as pd
import joblib
st.title("Loan Prediction Model")
st.write("Enter the details below to predict the result")
credit_score = st.number_input("Enter Credit Score")
income = st.number_input("Enter Income")
loan_amount = st.number_input("Enter Loan Amount")
if st.button("Predict"):
        model = joblib.load("model.pkl")
prediction = model.predict([[credit_score, income, loan_amount]])
st.success(f"Prediction: {prediction[0]}")
            