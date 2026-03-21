#Copyright@Zhenhui Yuan, 2025
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Create Dataset
network_data = {
    'packet_size': [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
    'latency': [10, 15, 20, 15, 10, 25, 30, 15, 50, 60],
    'packet_type': ['legitimate'] * 8 + ['malicious'] * 2
}
df = pd.DataFrame(network_data)
X = df[['packet_size', 'latency']]
y = df['packet_type']

# Print Original Dataset and Distribution
print("Original Dataset:")
print(df)
print("\nClass Distribution:")
print(y.value_counts())

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle Imbalance with SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Print Resampled Dataset and Distribution
resampled_df = pd.DataFrame(X_train_resampled, columns=['packet_size', 'latency'])
resampled_df['packet_type'] = y_train_resampled
print("\nResampled Dataset:")
print(resampled_df)
print("\nResampled Class Distribution:")
print(y_train_resampled.value_counts())
