#Copyright@Zhenhui Yuan, 2025
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Example Dataset
data = {
    'Bandwidth (Mbps)': [100, 500, 1000, 200],
    'Latency (ms)': [20, 50, 10, 80],
    'Packet Loss (%)': [0.1, 1.2, 0.3, 0.7]
}
df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Min-Max Scaling (Normalisation: scales data to [0, 1])
min_max_scaler = MinMaxScaler()
scaled_minmax = min_max_scaler.fit_transform(df)
df_minmax = pd.DataFrame(scaled_minmax, columns=df.columns)

# Standard Scaling (Standardisation: mean = 0, std = 1)
standard_scaler = StandardScaler()
scaled_standard = standard_scaler.fit_transform(df)
df_standard = pd.DataFrame(scaled_standard, columns=df.columns)

print("\nMin-Max Scaled Data:")
print(df_minmax)

print("\nStandard Scaled Data:")
print(df_standard)
