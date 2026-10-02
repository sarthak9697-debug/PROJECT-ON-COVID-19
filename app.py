import joblib
import streamlit as st

# 1. Decorate the loading function
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

# 2. Call the function to get the objects
model, scaler = load_artifacts()