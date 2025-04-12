# -*- coding: utf-8 -*-
"""GW2.ipynb
Original file is located at
    https://colab.research.google.com/drive/1kACBG7nebqOjqhS9_k7B8YIyzfTM20n4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

df_metrics = pd.read_csv("kubernetes_performance_metrics_dataset[1].csv")
df_resources = pd.read_csv("kubernetes_resource_allocation_dataset[1].csv")
print("Columns in df_metrics:", df_metrics.columns)
print("Columns in df_resources:", df_resources.columns)

print("Performance Metrics Dataset:")
print(df_metrics.info())
print(df_metrics.head())

print("Resource Allocation Dataset:")
print(df_resources.info())
print(df_resources.head())

print("Missing Values in Performance Metrics Dataset:")
print(df_metrics.isnull().sum())

print("\nMissing Values in Resource Allocation Dataset:")
print(df_resources.isnull().sum())

print("Summary Statistics (Performance Metrics Dataset):")
print(df_metrics.describe())

print("\nSummary Statistics (Resource Allocation Dataset):")
print(df_resources.describe())

common_columns = ['pod_name', 'namespace']
df = pd.merge(df_metrics, df_resources, on=common_columns, how='inner')

# Convert timestamp to datetime & sort
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', inplace=True)
else:
    print("Warning: No 'timestamp' column found!")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
# Use df instead of df_metrics
sns.histplot(df['cpu_usage'], kde=True, bins=30, color='blue')
plt.title("CPU Usage Distribution")

plt.subplot(1,2,2)
# Use df instead of df_metrics
sns.histplot(df['memory_usage'], kde=True, bins=30, color='red')
plt.title("Memory Usage Distribution")

plt.show()

plt.figure(figsize=(10,6))
# Exclude non-numeric columns before calculating correlations
numeric_df = df_metrics.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Performance Metrics Dataset)")
plt.show()

num_cols = ['cpu_usage', 'memory_usage', 'network_bandwidth_usage']
df_num = df[num_cols]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_num)
df_scaled = pd.DataFrame(df_scaled, columns=num_cols, index=df.index)

# **Anomaly Detection using One-Class SVM**
svm_model = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
df['anomaly_svm'] = svm_model.fit_predict(df_scaled)

# **Anomaly Detection using Isolation Forest**
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_iso'] = iso_forest.fit_predict(df_scaled)

# Convert -1 (anomaly) and 1 (normal) to 0/1
df['anomaly_svm'] = df['anomaly_svm'].apply(lambda x: 1 if x == -1 else 0)
df['anomaly_iso'] = df['anomaly_iso'].apply(lambda x: 1 if x == -1 else 0)

# **Plot CPU Usage Anomalies**
plt.figure(figsize=(12,6))
sns.scatterplot(x=df.index, y=df['cpu_usage'], hue=df['anomaly_svm'], palette={0:'blue', 1:'red'})
plt.title('CPU Usage Anomalies (SVM)')
plt.show()

plt.figure(figsize=(12,6))
sns.scatterplot(x=df.index, y=df['cpu_usage'], hue=df['anomaly_iso'], palette={0:'blue', 1:'red'})
plt.title('CPU Usage Anomalies (Isolation Forest)')
plt.show()

# Print anomaly summary
print("SVM Anomaly Distribution:")
print(df['anomaly_svm'].value_counts())
print("\nIsolation Forest Anomaly Distribution:")
print(df['anomaly_iso'].value_counts())

# Add anomaly scores as new features
df['anomaly_score'] = df['anomaly_svm'] + df['anomaly_iso']

# **LSTM Model for Forecasting**
features = num_cols + ['anomaly_score']
df_lstm = df[['timestamp'] + features].set_index('timestamp')

# Normalize data using MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_lstm)
df_scaled = pd.DataFrame(df_scaled, columns=features, index=df_lstm.index)

# Function to create sequences
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Create sequences
seq_length = 50
X, y = create_sequences(df_scaled.values, seq_length)

# Split into train & test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(features)))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], len(features)))

# **Optimized LSTM Model**
model = Sequential()

# First LSTM layer
model.add(LSTM(units=256, activation='relu', return_sequences=True, input_shape=(seq_length, len(features))))
model.add(Dropout(0.4))
model.add(BatchNormalization())

# Second LSTM layer
model.add(LSTM(units=64, activation='relu', return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

# Third LSTM layer
model.add(LSTM(units=48, activation='relu'))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(len(features)))

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', marker='o', linestyle='-')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss Curve')
plt.show()

# **Make Predictions**
y_pred = model.predict(X_test)

# **Inverse Transform Predictions**
y_test_inv = scaler.inverse_transform(np.hstack((y_test, np.zeros((y_test.shape[0], df_scaled.shape[1] - y_test.shape[1])))))[:, :len(num_cols)]
y_pred_inv = scaler.inverse_transform(np.hstack((y_pred, np.zeros((y_pred.shape[0], df_scaled.shape[1] - y_pred.shape[1])))))[:, :len(num_cols)]

# **Plot Actual vs Predicted Values**
plt.figure(figsize=(12,6))
plt.plot(df_lstm.index[train_size+seq_length:], y_test_inv[:, 0], label="Actual CPU Usage", color='blue')
plt.plot(df_lstm.index[train_size+seq_length:], y_pred_inv[:, 0], label="Predicted CPU Usage", color='red')
plt.xlabel("Timestamp", fontsize=12)
plt.ylabel("CPU Usage", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.title("Actual vs Predicted CPU Usage", fontsize=14)
plt.legend()
plt.show()

from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Assume X_train was used for training (2927, 50, 4)
scaler = MinMaxScaler()

# Fit on all 4 features
X_train_reshaped = X_train.reshape(-1, 4)  # Reshape to (total_samples, 4)
scaler.fit(X_train_reshaped)

# Save the updated scaler
import joblib
joblib.dump(scaler, "scaler.pkl")

import numpy as np
import joblib
from keras.models import load_model
import keras.losses

# Load models
svm_model = joblib.load("svm_model.pkl")
iso_forest = joblib.load("isolation_forest.pkl")
scaler = joblib.load("scaler.pkl")  # Make sure this scaler was fitted on 4 features
lstm_model = load_model("model.h5", custom_objects={"mse": keras.losses.MeanSquaredError()})

# Take input
cpu_usage = float(input("Enter CPU Usage (0 to 1): "))
memory_usage = float(input("Enter Memory Usage (0 to 1): "))
network_usage = float(input("Enter Network Usage (0 to 1): "))

# Add a placeholder for the missing 4th feature (set to 0 for now)
dummy_feature = np.zeros((1, 1))  # Adjust if you know what this should be
input_data = np.array([[cpu_usage, memory_usage, network_usage]])
input_data_expanded = np.hstack((input_data, dummy_feature))  # Now it has 4 features

# Scale input
input_data_scaled = scaler.transform(input_data_expanded)

# Reshape to match LSTM input shape (batch_size=1, time_steps=1, features=4)
input_data_reshaped = np.reshape(input_data_scaled, (1, 1, input_data_scaled.shape[1]))

# Make prediction
predicted_usage = lstm_model.predict(input_data_reshaped)

# Inverse transform the first 3 features
predicted_usage_trimmed = predicted_usage[:, :3]
# Add a placeholder for the missing feature (assuming it's 0 or mean)
missing_feature = np.zeros((predicted_usage_trimmed.shape[0], 1))  # Shape: (1,1)
predicted_usage_fixed = np.hstack([predicted_usage_trimmed, missing_feature])  # Shape: (1,4)

# Now apply inverse transform
predicted_usage_inv = scaler.inverse_transform(predicted_usage_fixed)

# Display results
print("\nPredicted Future Resource Usage:")
print(f"CPU Usage: {predicted_usage_inv[0][0]:.4f}")
print(f"Memory Usage: {predicted_usage_inv[0][1]:.4f}")
print(f"Network Usage: {predicted_usage_inv[0][2]:.4f}")

