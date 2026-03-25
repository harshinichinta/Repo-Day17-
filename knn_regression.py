import streamlit as st
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsRegressor  # <- important!

st.title("KNN Regression Prediction App")

# Load model safely with caching
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except ModuleNotFoundError as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

if model:
    # Input fields
    credit_score = st.number_input("Credit Score")
    income = st.number_input("Income")
    loan_amount = st.number_input("Loan Amount")

    # Prediction
    if st.button("Predict"):
        input_array = np.array([[credit_score, income, loan_amount]])
        prediction = model.predict(input_array)
        st.success(f"Prediction: {prediction[0]}")
else:
    st.warning("The model could not be loaded. Please check your environment and ensure 'model.pkl' exists.")