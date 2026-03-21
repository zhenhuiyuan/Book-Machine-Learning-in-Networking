#Copyright@Zhenhui Yuan, 2025

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# load the dataset
dataset_path = "../../../Dataset/Network Intrusion Detection_UNSW-NB 15/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv"  # Replace with the correct file path
data = pd.read_csv(dataset_path)

print("Dataset Shape:", data.shape)
print("Column Names:", data.columns)
print(data.info())

# Step 2: Preprocess the Dataset
# Drop irrelevant columns (e.g., 'id')
data.drop(['id'], axis=1, inplace=True)

# Encode categorical columns using LabelEncoder
categorical_columns = ['proto', 'service', 'state', 'attack_cat']
label_encoder = LabelEncoder()

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])


# Define features (X) and labels (y)
X = data.drop(['label'], axis=1)  # Features
y = data['label']  # Labels (0: Normal, 1: Attack)

# Check class distribution in the entire dataset
print("\nClass distribution in the dataset:")
print(y.value_counts())

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
#stratify=y ensure both classes are present in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print class distribution in the training dataset
print("\nClass distribution in the training dataset:")
print(y_train.value_counts())

# Initialize Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Check class distribution in the test set (prediction vs. ground truth)
print("\nTest Set Prediction Distribution:")
print(np.bincount(y_pred))

# Evaluate performance
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
