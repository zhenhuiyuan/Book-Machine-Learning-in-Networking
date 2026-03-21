#Copyright@Zhenhui Yuan, 2025
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score

# Step 1:Load KDD99 dataset
# Load the dataset
url = "../../../Dataset/KDD99/kddcup.data_10_percent.gz"
data = pd.read_csv(url, header=None)

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
data.columns = column_names
# Load dataset
df = pd.read_csv(url, names=column_names, compression='gzip')

# Drop non-numeric columns (e.g., protocol_type, service, flag)
df_numeric = df.select_dtypes(include=[np.number])

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

print(f"Dataset Shape: {df_numeric.shape}")

#Step 2: Apply k - Means

# Apply k-means with k=2 (Normal vs Attack)
k = 2
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_numeric['cluster'] = kmeans.fit_predict(X_scaled)

# Check cluster distribution
print(df_numeric['cluster'].value_counts())

# metric 1: Compute inertia (Elbow method idea), Measures how tight the clusters are.
print(f"Inertia: {kmeans.inertia_}")

# metric 2: checking how distinct clusters are.
# A higher score (close to 1) means better-defined clusters.
# A score close to 0 means overlapping clusters.
# A negative score means incorrect clustering.
#sil_score = silhouette_score(X_scaled, df_numeric['cluster'])
#print(f"Silhouette Score: {sil_score}")

# metric 3:Adjusted Rand Index (ARI)
# how similar the clustering is to actual labels.
# 0 (Random Clustering): No correlation between predicted and true labels.
# 1 (Perfect Clustering): Perfect match with the ground truth labels.
# Assuming 'label' column has ground truth labels
ari = adjusted_rand_score(df['label'], df_numeric['cluster'])
print(f"Adjusted Rand Index: {ari}")

#Step 3: Visualization using PCA
# Reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_numeric['cluster'], cmap='coolwarm', alpha=0.5)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering on KDD99 (Network Traffic)")
plt.colorbar(label="Cluster ID")
plt.show()
