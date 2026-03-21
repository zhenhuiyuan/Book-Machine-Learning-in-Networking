#Copyright@Zhenhui Yuan, 2025
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load UNSW-NB15 Dataset
# Replace 'your_path_to_dataset' with the path to the dataset file
dataset_path = "../../Dataset/Network Intrusion Detection_UNSW-NB 15/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv"  # Replace with the correct file path
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

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: 5-Fold Cross-Validation
#The model is trained on 4 folds and validated on the remaining fold, repeating for all folds.
kf = KFold(n_splits=5, shuffle=True, random_state=42)
#A Random Forest Classifier is used for simplicity, but you can replace it with any model suitable for networking tasks.
model = RandomForestClassifier(random_state=42)

# Evaluate model using cross-validation: calculates accuracy for each fold.
# You do not need to manually call model.fit() when using cross_val_score. The cross_val_score function from sklearn
# takes care of splitting the data, training the model on the training folds, and evaluating it on the validation fold internally.
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Step 4: Output Results
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# ***** try with different value of k *****
# Test different values of k (number of folds)
k_values = [2, 5, 10, 20]  # You can add more values as needed
results = {}

for k in k_values:
    print(f"Testing k = {k} folds...")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    results[k] = {
        "mean_accuracy": np.mean(cv_scores),
        "std_dev": np.std(cv_scores),
    }
    print(f"k = {k}: Mean Accuracy = {np.mean(cv_scores):.4f}, Std Dev = {np.std(cv_scores):.4f}")

# Output Optimal k
optimal_k = max(results, key=lambda k: results[k]["mean_accuracy"])
print("\nOptimal k value based on Mean Accuracy:")
print(f"k = {optimal_k}: {results[optimal_k]}")
