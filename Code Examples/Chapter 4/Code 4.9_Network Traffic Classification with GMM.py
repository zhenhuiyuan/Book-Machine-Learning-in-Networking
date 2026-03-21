#Copyright@Zhenhui Yuan, 2025
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd

# Simulated network traffic data (packet size, inter-arrival time)
# VoIP: 500 → Average packet size (bytes), 20 → Average inter-packet arrival time (ms)
#  the number of samples to generate: 100
np.random.seed(42)
traffic_data = np.concatenate([
    np.random.multivariate_normal([500, 20], [[20000, 10], [10, 5]], 100),  # VoIP
    np.random.multivariate_normal([1200, 50], [[30000, 15], [15, 10]], 100), # Streaming
    np.random.multivariate_normal([800, 30], [[25000, 20], [20, 8]], 100)   # Web browsing
])

# Convert to DataFrame
df = pd.DataFrame(traffic_data, columns=["Packet Size (bytes)", "Inter-Arrival Time (ms)"])

# Apply Gaussian Mixture Model (GMM) for clustering
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
df['Cluster'] = gmm.fit_predict(df[['Packet Size (bytes)', 'Inter-Arrival Time (ms)']])

# Visualize Results
plt.figure(figsize=(8,6))
for cluster in range(3):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Packet Size (bytes)'], cluster_data['Inter-Arrival Time (ms)'], label=f'Cluster {cluster}')

plt.xlabel("Packet Size (bytes)")
plt.ylabel("Inter-Arrival Time (ms)")
plt.title("Network Traffic Clustering using GMM")
plt.legend()
plt.grid()
plt.show()
