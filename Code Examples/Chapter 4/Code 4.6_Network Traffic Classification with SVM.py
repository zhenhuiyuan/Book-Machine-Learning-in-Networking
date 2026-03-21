#Copyright@Zhenhui Yuan, 2025

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt

# Load the dataset
url = "kddcup.data_10_percent.gz"
data = pd.read_csv(url, header=None)

# Add column names
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]
data.columns = columns

# Convert categorical data to numerical data
label_encoder = LabelEncoder()
data['protocol_type'] = label_encoder.fit_transform(data['protocol_type'])
data['service'] = label_encoder.fit_transform(data['service'])
data['flag'] = label_encoder.fit_transform(data['flag'])

# Simplify the problem by classifying as normal or attack
data['label'] = data['label'].apply(lambda x: 0 if x == 'normal.' else 1)

# Features and labels
X = data.drop('label', axis=1)
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#SVM performs better when the features are scaled.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#The KDD Cup 1999 dataset has 41 features. Reduce dimensionality using PCA.
pca = PCA(n_components=10)  # Reduce to 10 components
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Start timer
start_time = time.time()

#Train the SVM Model
# Create an SVM classifier
svm_classifier = SVC(kernel='linear')
# Train the model
svm_classifier.fit(X_train, y_train)

# Calculate elapsed time
elapsed_time = time.time() - start_time
print(f"Training completed in {elapsed_time:.2f} seconds")

# Evaluate the Model
# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
