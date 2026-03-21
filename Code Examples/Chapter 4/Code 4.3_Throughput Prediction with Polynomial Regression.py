#Copyright@Zhenhui Yuan, 2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
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
throughput = 0.9 * bandwidth - 0.3 * latency + 0.002 * packet_size + np.random.normal(0, 5, n_samples)

# Create DataFrame
df = pd.DataFrame({"Bandwidth": bandwidth, "Latency": latency, "PacketSize": packet_size, "Throughput": throughput})

# Select features and target variable
X = df[["Bandwidth", "Latency", "PacketSize"]]
y = df["Throughput"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial degrees to test
degrees = [2, 4, 6, 8]
colors = ["red", "green", "blue", "purple"]

plt.figure(figsize=(8, 6))

# Train and evaluate models for each polynomial degree
for degree, color in zip(degrees, colors):
    # Apply Polynomial Feature Transformation
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train polynomial regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Make predictions
    y_pred = model.predict(X_test_poly)

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Polynomial Regression (Degree {degree}):")
    print(f"  - Mean Squared Error (MSE): {mse:.2f}")
    print(f"  - R² Score: {r2:.4f}")
    print(f"  - Number of Features After Transformation: {X_train_poly.shape[1]}\n")

    # Plot actual vs predicted throughput
    plt.scatter(y_test, y_pred, color=color, label=f"Degree {degree} (R²: {r2:.2f})", alpha=0.6)

# Plot perfect prediction line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "k--", lw=2,
         label="Perfect Prediction")  # Diagonal line
plt.xlabel("Actual Throughput (Mbps)")
plt.ylabel("Predicted Throughput (Mbps)")
plt.title("Polynomial Regression: Actual vs Predicted Throughput")
plt.legend()
plt.grid(True)
plt.show()
