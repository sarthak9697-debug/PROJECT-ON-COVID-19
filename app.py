import streamlit as st
import joblib
import pandas as pd
from prophet import Prophet

st.set_page_config(page_title="Prophet Forecast", layout="centered")

# 1. Load the Prophet model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Time Series Forecasting with Prophet")

# 2. Input: Number of periods (days) to forecast
periods = st.slider("Select forecast horizon (days):", min_value=7, max_value=365, value=30)

if st.button("Generate Forecast"):
    # Prophet requires future dates dataframe
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    # Display the latest predictions
    st.subheader("Forecast Results")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods))
    
    # Plot forecast
    fig = model.plot(forecast)
    st.pyplot(fig)