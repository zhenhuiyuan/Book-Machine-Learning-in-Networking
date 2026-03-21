#Copyright@Zhenhui Yuan, 2025

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

# Generate synthetic network dataset
np.random.seed(42)
n_samples = 1000

# Features: Bandwidth, Packet Loss, Throughput, Jitter
X = np.random.rand(n_samples, 4) * [100, 5, 1000, 50]  # Scaling features

# Target Variables
y_class = np.random.choice(["Low", "Medium", "High"], size=n_samples)  # Congestion Level (Classification)
y_reg = np.random.rand(n_samples) * 100  # Latency in ms (Regression)

# Encode categorical labels
y_class_encoded = pd.factorize(y_class)[0]  # Convert labels to numeric

# Train-test split
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class_encoded, y_reg, test_size=0.2, random_state=42
)

# Create individual models (use RandomForestRegressor for regression)
classifier = RandomForestClassifier(n_estimators=50, random_state=42)  # For classification task
regressor = RandomForestRegressor(n_estimators=50, random_state=42)  # For regression task

# Train models separately
classifier.fit(X_train, y_class_train)
regressor.fit(X_train, y_reg_train)

# Predictions
y_class_pred = classifier.predict(X_test)
y_reg_pred = regressor.predict(X_test)

# Evaluate individual models
class_acc = accuracy_score(y_class_test, y_class_pred)
reg_mae = mean_absolute_error(y_reg_test, y_reg_pred)

print(f"Classification Accuracy: {class_acc:.2f}")
print(f"Regression MAE: {reg_mae:.2f} ms")

# --- STACKING ENSEMBLE: Combine Both Models ---
# Meta-model (use a regressor, since we are dealing with a regression task)
meta_model = RandomForestRegressor(n_estimators=50, random_state=42)

# Stacking Model (Use both RandomForestRegressors in the stacking)
stacking_model = StackingRegressor(
    estimators=[("rf_reg_1", regressor), ("rf_reg_2", regressor)],  # Use regressors only
    final_estimator=meta_model
)

# Train stacking model
stacking_model.fit(X_train, y_reg_train)  # We train it for regression task

# Predict with stacking model
y_reg_stacked_pred = stacking_model.predict(X_test)
stacked_mae = mean_absolute_error(y_reg_test, y_reg_stacked_pred)

print(f"Stacked Regression MAE: {stacked_mae:.2f} ms")
