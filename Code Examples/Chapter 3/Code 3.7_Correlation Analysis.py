#Copyright@Zhenhui Yuan, 2025
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate network data
np.random.seed(42)
time_index = pd.date_range(start="2025-01-01", periods=100, freq="T")  # 100 minutes of data
latency = np.random.normal(loc=50, scale=5, size=100)  # Simulated latency in ms with noise

# Create a DataFrame
network_data = pd.DataFrame({'Timestamp': time_index, 'Latency_ms': latency})

# Introduce a spike for anomaly detection
network_data.loc[30:40, 'Latency_ms'] += 30

# Temporal analysis
# 1. Plot time-series data
plt.figure(figsize=(10, 5))
sns.lineplot(x='Timestamp', y='Latency_ms', data=network_data, label='Latency')
plt.title('Network Latency Over Time')
plt.xlabel('Time')
plt.ylabel('Latency (ms)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# 2. Detect anomalies
threshold = 70  # Define a threshold for high latency
anomalies = network_data[network_data['Latency_ms'] > threshold]
print("Detected Anomalies:")
print(anomalies)

# 3. Rolling average for trend analysis
network_data['Rolling_Avg'] = network_data['Latency_ms'].rolling(window=5).mean()

# Plot rolling average
plt.figure(figsize=(10, 5))
sns.lineplot(x='Timestamp', y='Latency_ms', data=network_data, label='Latency')
sns.lineplot(x='Timestamp', y='Rolling_Avg', data=network_data, label='5-min Rolling Average', color='orange')
plt.title('Latency with Rolling Average')
plt.xlabel('Time')
plt.ylabel('Latency (ms)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
