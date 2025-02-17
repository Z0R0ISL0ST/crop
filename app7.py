import numpy as np
import pickle
import pandas as pd
import streamlit as st

# Load models
models = {
    "Decision Tree": pickle.load(open("crop_pred_dtree.pkl", "rb")),
    "Random Forest": pickle.load(open("crop_pred_rand.pkl", "rb")),
    "K-Nearest Neighbors (KNN)": pickle.load(open("crop_pred_knn.pkl", "rb")),
    "Support Vector Machine (SVM)": pickle.load(open("crop_pred_svc.pkl", "rb"))
}

# Streamlit UI
st.title("Crop Prediction App")
st.markdown("### Select a model and enter input values")

# Sidebar model selection
model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))

# Input fields
temperature = st.number_input("Temperature", min_value=0.0, format="%.2f")
humidity = st.number_input("Humidity", min_value=0.0, format="%.2f")
ph = st.number_input("pH", min_value=0.0, format="%.2f")

# Prediction
def predict_crop(model, features):
    return model.predict([features])[0]

if st.button("Predict Crop"):
    model = models[model_choice]
    result = predict_crop(model, [temperature, humidity, ph])
    st.success(f"Predicted Crop: {result}")
