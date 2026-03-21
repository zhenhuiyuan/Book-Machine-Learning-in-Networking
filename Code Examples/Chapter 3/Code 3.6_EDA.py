#Copyright@Zhenhui Yuan, 2025
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset (update the path with the actual file location)
dataset_path = "../../../Dataset/Network Intrusion Detection_UNSW-NB 15/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv"  # Replace with the correct file path
data = pd.read_csv(dataset_path)

# Display basic information
print("Dataset Shape:", data.shape)
print("Column Names:", data.columns)
print(data.info())

#The sum of ’synack’ and ’ackdat’ of the TCP. You can replace with other features.
feature = 'tcprtt'

print(f"\nDescriptive Statistics for {feature}:")
print("Mean:", data[feature].mean())
print("Median:", data[feature].median())
print("Mode:", data[feature].mode().iloc[0])  # Mode may return multiple rows
print("Variance:", data[feature].var())
print("Standard Deviation:", data[feature].std())
print("Minimum:", data[feature].min())
print("Maximum:", data[feature].max())
print("25th Percentile:", data[feature].quantile(0.25))
print("50th Percentile (Median):", data[feature].quantile(0.50))
print("75th Percentile:", data[feature].quantile(0.75))

# Histogram for Numerical Features
# x-axis is divided into intervals (bins), each bin represents a range of values from the data.
# y-axis shows the frequency (count) of data points in each bin
plt.figure(figsize=(8, 4))
sns.histplot(data[feature], kde=True, bins=50)
plt.title(f"Distribution of {feature}")
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.show()

# Boxplot for Outlier Detection
# visualizing using a box and whiskers
# box: the median (50th percentile) of the data. The box itself spans from the lower quartile (Q1, 25th percentile) to the upper quartile (Q3, 75th percentile).
# whiskers: Extend to include most data points within 1.5 times the IQR.
# Data points outside the whiskers are plotted as individual points, representing outliers.
plt.figure(figsize=(8, 4))
sns.boxplot(data[feature])
plt.title(f"Boxplot of {feature}")
plt.xlabel(feature)
plt.show()

# Scatter Plot for Relationship Between Two Variables
# sbytes: Number of bytes sent by the source (initiator) in a network session.
# dbytes: Number of bytes received by the destination in the network session.
# attack_cat: Category of the network attack. This dataset includes nine categories (e.g., Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode and Worms)
# normal: a larger upload (sbytes) can lead to a larger response (dbytes).
# under-attack: a DoS attack might show unusual patterns like a high sbytes with minimal dbytes. Different attack categories (attack_cat) may show distinct patterns between sbytes and dbytes
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['sbytes'], y=data['dbytes'], hue=data['attack_cat'], alpha=0.6)
plt.title("Scatter Plot of Source Bytes vs Destination Bytes")
plt.xlabel("Source Bytes")
plt.ylabel("Destination Bytes")
plt.legend(title="Attack Category", loc="upper right")
plt.show()

# Heatmap for Correlations (with selected features for clarity)
# Select numeric features
numeric_data = data.select_dtypes(include=[np.number])

# Option 1: Manually select a subset of important features
selected_features = ['sbytes', 'dbytes', 'sload', 'dload', 'sttl', 'dttl', 'ct_state_ttl', 'tcprtt', 'ackdat', 'synack']

# Ensure all selected features are in the dataset
selected_features = [feat for feat in selected_features if feat in numeric_data.columns]

# Compute the correlation matrix of selected features
correlation_matrix = numeric_data[selected_features].corr()

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap of Selected Features")
plt.tight_layout()
plt.show()


# Violin plot
violin_feature = 'sloss'  # Replace with your desired feature
category = 'attack_cat'  # Use attack categories as a grouping variable

# Ensure the selected feature and category are in the dataset
if violin_feature in data.columns and category in data.columns:
    # Create a violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=category, y=violin_feature, data=data, palette="muted", scale="count", bw=0.2, cut=0, inner="box")
    plt.title(f"Violin Plot of {violin_feature} by {category}")
    plt.xlabel("Attack Category")
    plt.ylabel(f"{violin_feature.capitalize()} (Values)")
    plt.xticks(rotation=45)
    plt.show()
else:
    print(f"Feature '{violin_feature}' or category '{category}' not found in dataset.")
