import streamlit as st
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsRegressor  # Needed to load the model

# ----------------------------
# App Title
# ----------------------------
st.title("KNN Regression Prediction App")

# ----------------------------
# Load model safely with caching
# ----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except FileNotFoundError:
        st.error("The file 'model.pkl' was not found. Make sure it exists in the same folder.")
        return None
    except ModuleNotFoundError as e:
        st.error(f"Model loading failed: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error while loading the model: {e}")
        return None

model = load_model()

# ----------------------------
# Input fields & Prediction
# ----------------------------
if model:
    credit_score = st.number_input("Credit Score", min_value=0)
    income = st.number_input("Income", min_value=0.0, format="%.2f")
    loan_amount = st.number_input("Loan Amount", min_value=0.0, format="%.2f")

    if st.button("Predict"):
        input_array = np.array([[credit_score, income, loan_amount]])
        prediction = model.predict(input_array)
        st.success(f"Predicted Value: {prediction[0]}")
else:
    st.warning("The model could not be loaded. Please check your environment and ensure 'model.pkl' exists.")