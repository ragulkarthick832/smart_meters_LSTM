import streamlit as st
import pickle
import numpy as np

# Load the trained models & scalers
with open("lstm_models.pkl", "rb") as f:
    data = pickle.load(f)

models = data["models"]
scalers = data["scalers"]

# Streamlit UI
st.set_page_config(page_title="LSTM Time Series Prediction", layout="centered")
st.title("📊 LSTM Time Series Prediction")
st.write("Predict daily average sensor values.")

# Dropdowns for Metric and Time Horizon
metric = st.selectbox("Select Metric:", ["Flow Rate (mL/sec)", "Current (mA)"])
time_horizon = st.selectbox("Select Time Horizon:", ["24hr", "1week", "1month", "3month"])

# Define the number of days
time_mapping = {"24hr": 1, "1week": 7, "1month": 30, "3month": 90}
days_to_predict = time_mapping[time_horizon]

# Prediction Button
if st.button("Predict"):
    model = models[metric]  # Load selected metric's model
    scaler = scalers[metric]  # Load scaler

    # Generate initial input (last 7 days of normalized values)
    X_input = np.random.rand(7, 1)  # Use random data (replace with real-time input if available)
    X_input = X_input.reshape(1, 7, 1)

    # Generate predictions
    predictions = []
    for _ in range(days_to_predict):
        pred = model.predict(X_input)
        predictions.append(pred[0][0])
        
        # Shift input for next prediction
        X_input = np.roll(X_input, -1)
        X_input[0, -1, 0] = pred[0][0]

    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Display Predictions
    st.success(f"Predicted {metric} for the next {days_to_predict} days:")
    st.write(predictions)
    message = "Performance Analysis: \nFlow Rate (mL/sec) RMSE: 0.2223\nCurrent (mA) RMSE: 0.2475"
    st.write(message)
# Run API: `streamlit run app.py`
