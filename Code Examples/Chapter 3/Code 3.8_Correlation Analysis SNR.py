#Copyright@Zhenhui Yuan, 2025
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Simulated dataset: SNR (in dB) and corresponding data rates (in Mbps)
np.random.seed(42)

# Generate SNR values (10 to 30 dB, simulating good signal quality)
snr = np.random.uniform(10, 30, 100)

# Generate data rate with some noise added (e.g., influenced by real-world factors like interference)
data_rate = 5 * np.log2(1 + snr) + np.random.normal(0, 0.5, len(snr))

# Create a DataFrame for better handling
df = pd.DataFrame({'SNR (dB)': snr, 'Data Rate (Mbps)': data_rate})

# Compute Pearson correlation coefficient
correlation, p_value = pearsonr(df['SNR (dB)'], df['Data Rate (Mbps)'])

# Display the dataset
print(df.head())

# Plot the data
plt.scatter(df['SNR (dB)'], df['Data Rate (Mbps)'], alpha=0.7, color='blue')
plt.title(f"SNR vs. Data Rate (Pearson r = {correlation:.2f})")
plt.xlabel('SNR (dB)')
plt.ylabel('Data Rate (Mbps)')
plt.grid(True)
plt.show()

# Print correlation result
print(f"Pearson Correlation Coefficient: {correlation:.2f}")
print(f"P-value: {p_value:.2e}")
