import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Title
st.title("KNN Classification using Iris Dataset")

# Load Dataset
data = load_iris()

x = data.data
y = data.target

# Dataset Information
st.subheader("Dataset")

df = pd.DataFrame(x, columns=data.feature_names)
df['Target'] = y

st.dataframe(df)

# Sidebar Inputs
st.sidebar.header("Model Parameters")

k = st.sidebar.slider("Select K Value", 1, 15, 3)

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=test_size,
    random_state=42
)

# Create Model
knn_classifier = KNeighborsClassifier(
    n_neighbors=k,
    metric='minkowski',
    p=2
)

# Train Model
knn_classifier.fit(x_train, y_train)

# Prediction
y_pred = knn_classifier.predict(x_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Display Results
st.subheader("Model Performance")

st.write(f"Accuracy : {accuracy:.2f}")
st.write(f"Precision : {precision:.2f}")
st.write(f"Recall : {recall:.2f}")
st.write(f"F1 Score : {f1:.2f}")

# Classification Report
st.subheader("Classification Report")

report = classification_report(y_test, y_pred)
st.text(report)

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

st.write(cm)

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 5))

cax = ax.matshow(cm, cmap='Blues')
plt.colorbar(cax)

# Add values inside matrix
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, f'{val}', ha='center', va='center', color='black')

# Labels
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

# Tick labels
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])

ax.set_xticklabels(data.target_names)
ax.set_yticklabels(data.target_names)

st.pyplot(fig)