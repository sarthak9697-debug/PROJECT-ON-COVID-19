import streamlit as st
import joblib
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(
    page_title="COVID-19 Case Forecaster",
    page_icon="🦠",
    layout="wide"
)

st.title("🦠 COVID-19 Future Case Forecaster")
st.write("Generate future case predictions using a trained Prophet time-series model.")

# 2. Cached Model Loading
@st.cache_resource
def load_model():
    # Make sure 'model.pkl' is placed in the same directory as app.py
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. User Controls (Sidebar)
st.sidebar.header("Forecast Settings")
periods = st.sidebar.slider(
    "Select forecast horizon (days):",
    min_value=7,
    max_value=180,
    value=30,
    step=1
)

# 4. Generate Predictions & Render UI
if st.button("Generate Forecast", type="primary"):
    with st.spinner("Calculating forecast..."):
        # Create future dataframe and predict
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # Slice the forecast window and rename columns
        results = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).copy()
        
        # Format the date and round values
        results["ds"] = pd.to_datetime(results["ds"]).dt.strftime("%Y-%m-%d")
        results["yhat"] = results["yhat"].round().astype(int)
        results["yhat_lower"] = results["yhat_lower"].round().astype(int)
        results["yhat_upper"] = results["yhat_upper"].round().astype(int)

        # Rename to user-friendly titles
        results = results.rename(
            columns={
                "ds": "Date",
                "yhat": "Predicted Cases",
                "yhat_lower": "Min Expected Cases",
                "yhat_upper": "Max Expected Cases",
            }
        )

        # Key Metrics Overview (for the final forecasted day)
        last_row = results.iloc[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Forecasted Date", last_row["Date"])
        col2.metric("Predicted Cases", f"{last_row['Predicted Cases']:,}")
        col3.metric("Expected Range", f"{last_row['Min Expected Cases']:,} - {last_row['Max Expected Cases']:,}")

        # Display Data Table
        st.subheader("📋 Forecast Data Table")
        st.dataframe(results, use_container_width=True)

        # Forecast Visualization Plot
        st.subheader("📈 Trend & Forecast Plot")
        fig = model.plot(forecast)
        st.pyplot(fig)