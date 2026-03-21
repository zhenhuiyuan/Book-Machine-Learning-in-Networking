#Copyright@Zhenhui Yuan, 2025
# results: F1 score, precision, recall are all 1, means the logistic regression model is perfect.
# precision of 1: every time the model predict 'good', it was correct. no false positive
# The logistic regression model found a perfect separation between good and bad classes. Since the dataset is
# very simple, clearly, psnr over 30 is good, below 30 is bad.
# The curve starts low for bad quality (0), rises steeply, and levels off for good quality (1)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

import numpy as np

import matplotlib.pyplot as plt

# Generate synthetic dataset of PSNR values

np.random.seed(42)
# generate 100 psnr samples
psnr_values = np.random.uniform(10, 50, 100)  # PSNR values range from 10 to 50 dB

threshold = 30  # Threshold for good/bad quality

labels = (psnr_values >= threshold).astype(int)  # Binary labels (1 for good, 0 for bad)

# Normalize PSNR values

mean_psnr = np.mean(psnr_values)

std_psnr = np.std(psnr_values)

normalized_psnr = (psnr_values - mean_psnr) / std_psnr

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(

    normalized_psnr.reshape(-1, 1), labels, test_size=0.2, random_state=42

)

# Train a logistic regression model

log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

# Make predictions on the test set

y_pred = log_reg.predict(X_test)

y_prob = log_reg.predict_proba(X_test)[:, 1]  # Predicted probabilities for class 1 (good quality)

# Evaluate the model

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred)

# Display the results

print(f"Accuracy: {accuracy:.2f}\n")

print("Classification Report:")

print(report)

# Plot the sigmoid curve and data

# Sort values for smooth curve plotting

sorted_indices = np.argsort(normalized_psnr)

sorted_psnr = normalized_psnr[sorted_indices]

sorted_labels = labels[sorted_indices]

# Compute sigmoid probabilities using the trained model

sigmoid_probs = log_reg.predict_proba(sorted_psnr.reshape(-1, 1))[:, 1]

plt.figure(figsize=(10, 6))

plt.scatter(psnr_values, labels, color='black', label='Original Data (PSNR)', alpha=0.6)

plt.plot(psnr_values[sorted_indices], sigmoid_probs, color='red', label='Sigmoid Curve')

plt.axvline(x=threshold, color='blue', linestyle='--', label=f'Threshold ({threshold} dB)')

plt.title("PSNR vs. Video Quality Classification using Logistic Regression")

plt.xlabel("PSNR (dB)")

plt.ylabel("Probability of Good Quality")

plt.legend()

plt.grid()

plt.show()

