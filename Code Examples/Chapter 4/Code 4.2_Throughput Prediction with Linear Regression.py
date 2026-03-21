#Copyright@Zhenhui Yuan, 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42)
n_samples = 200  # Number of data points

# Simulating real-world network conditions
bandwidth = np.random.uniform(10, 100, n_samples)  # Bandwidth in Mbps
latency = np.random.uniform(5, 100, n_samples)  # Latency in ms
packet_size = np.random.uniform(500, 1500, n_samples)  # Packet size in bytes

# Define throughput using a realistic formula with some noise
# The formula provides the groundtruth labels.
throughput = 0.9 * bandwidth - 0.3 * latency + 0.002 * packet_size + np.random.normal(0, 5, n_samples)

# Create DataFrame
df = pd.DataFrame({"Bandwidth": bandwidth, "Latency": latency, "PacketSize": packet_size, "Throughput": throughput})

# Select features and target variable
X = df[["Bandwidth", "Latency", "PacketSize"]]
y = df["Throughput"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Plot actual vs predicted throughput
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="red", label="Predicted Throughput", alpha=0.6)
plt.scatter(y_test, y_test, color="blue", label="Actual Throughput", alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "k--", lw=2, label="Perfect Prediction")  # Diagonal reference line
plt.xlabel("Actual Throughput (Mbps)")
plt.ylabel("Predicted Throughput (Mbps)")
plt.title("Actual vs. Predicted Throughput")
plt.legend()
plt.grid(True)
plt.show()
