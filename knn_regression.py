import streamlit as st
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

st.title("KNN Regression App")

# Sample dataset (you can change later)
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])

# Train model
model = KNeighborsRegressor(n_neighbors=2)
model.fit(X, y)

# User input
value = st.number_input("Enter a value")

# Prediction
if st.button("Predict"):
    prediction = model.predict([[value]])
    st.success(f"Prediction: {prediction[0]}")