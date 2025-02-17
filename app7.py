import numpy as np
import pickle
import pandas as pd
import streamlit as st
import gdown

# Google Drive Model URLs
dtree_url = "https://drive.google.com/uc?id=1XhOm4mtO2xuoOln4Kug7PZZj3y9FeK8v"
rand_url = "https://drive.google.com/uc?id=1HxhlV3ScGPX7JW5sQ6bn3Js97zGP_rkc"
knn_url = "https://drive.google.com/uc?id=16WePgQYxSTZYASn6eMptJVgPQIlY1NBF"
svc_url = "https://drive.google.com/uc?id=YOUR_SVM_MODEL_ID"

# Output filenames
dtree_output = "crop_pred_dtree.pkl"
rand_output = "crop_pred_rand.pkl"
knn_output = "crop_pred_knn.pkl"
svc_output = "crop_pred_svc.pkl"

# Download models from Google Drive
gdown.download(dtree_url, dtree_output, quiet=False)
gdown.download(rand_url, rand_output, quiet=False)
gdown.download(knn_url, knn_output, quiet=False)
gdown.download(svc_url, svc_output, quiet=False)

# Load models
models = {
    "Decision Tree": pickle.load(open(dtree_output, "rb")),
    "Random Forest": pickle.load(open(rand_output, "rb")),
    "K-Nearest Neighbors (KNN)": pickle.load(open(knn_output, "rb")),
    "Support Vector Machine (SVM)": pickle.load(open(svc_output, "rb"))
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
