import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("new_sensor_data.csv")

# Convert Timestamp column to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H-%M-%S")

# Sort data by time
df = df.sort_values("Timestamp")

# Aggregate daily averages
df["Date"] = df["Timestamp"].dt.date
daily_avg = df.groupby("Date").mean().reset_index()

# Normalize values
scalers = {}  # Store scalers for each metric
for metric in ["Flow Rate (mL/sec)", "Current (mA)"]:
    scaler = MinMaxScaler()
    daily_avg[metric] = scaler.fit_transform(daily_avg[[metric]])
    scalers[metric] = scaler  # Store the scaler

# Prepare data for LSTM
def prepare_data(series, time_steps=7):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i : i + time_steps])
        y.append(series[i + time_steps])
    return np.array(X), np.array(y)

# Train separate models for each metric
models = {}
history_logs = {}

for metric in ["Flow Rate (mL/sec)", "Current (mA)"]:
    print(f"Training LSTM model for: {metric}")

    data_series = daily_avg[metric].values
    X, y = prepare_data(data_series, time_steps=7)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM

    # Build LSTM Model
    model = Sequential([
        LSTM(50, activation="relu", return_sequences=True, input_shape=(7, 1)),
        LSTM(50, activation="relu"),
        Dense(1)
    ])
    
    model.compile(optimizer="adam", loss="mse")
    
    # Train model
    history = model.fit(X, y, epochs=50, batch_size=8, verbose=1)
    
    # Store trained model
    models[metric] = model
    history_logs[metric] = history.history["loss"]

# Save models & scalers using Pickle
with open("lstm_models.pkl", "wb") as f:
    pickle.dump({"models": models, "scalers": scalers}, f)

# Plot training loss
plt.figure(figsize=(8, 5))
for metric in history_logs:
    plt.plot(history_logs[metric], label=metric)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss for LSTM Models")
plt.show()

# Performance Analysis
print("\nPerformance Analysis:")

for metric in ["Flow Rate (mL/sec)", "Current (mA)"]:
    data_series = daily_avg[metric].values
    X, y = prepare_data(data_series, time_steps=7)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    predictions = models[metric].predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    
    print(f"{metric} RMSE: {rmse:.4f}")

print("LSTM training completed. Models saved to lstm_models.pkl")
