#Copyright@Zhenhui Yuan, 2025
#The trained model predicts whether network traffic is an attack or normal.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# load the dataset
df = pd.read_csv("../../Dataset/KDD99/kddcup.data_10_percent.gz", header=None)

# Add column names
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]
df.columns = column_names

# Selecting important columns (dropping non-numeric categorical data)
numeric_features = ["duration", "src_bytes", "dst_bytes", "wrong_fragment",
    "count", "srv_count", "same_srv_rate", "dst_host_same_srv_rate"]  # Example features
# Extract selected numeric features and target label
X = df[numeric_features]
y = df["label"]

# Encode labels (attack types)
le = LabelEncoder()
y = le.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(clf, "decision_tree_model.pkl")

# Load the model later for inference
loaded_model = joblib.load("decision_tree_model.pkl")

# Example: Predict on new incoming data
new_data = [[0, 500, 1000, 0, 10, 20, 0.9, 0.8]]  # Example feature values
prediction = loaded_model.predict(new_data)

# Convert numeric prediction back to original attack type
predicted_attack_type = le.inverse_transform([prediction[0]])
print("Predicted Attack Type:", predicted_attack_type[0])
