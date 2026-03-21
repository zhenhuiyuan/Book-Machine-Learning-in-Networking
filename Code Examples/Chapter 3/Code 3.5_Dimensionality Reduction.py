#Copyright@Zhenhui Yuan, 2025
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
sns.set(style='darkgrid')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os

# ********* Loading the dataset *********
# Replace with your actual path
#dataset_path = r"/Users/u2273078/Documents/Research/Machine Learning/ML for Communications and Networking/Dataset/Network Intrusion Detection_CIC-IDS-2017/Monday-WorkingHours.pcap_ISCX.csv"
data1 = pd.read_csv('../../../Dataset/Network Intrusion Detection_CIC-IDS-2017/Monday-WorkingHours.pcap_ISCX.csv')
data2 = pd.read_csv('../../../Dataset/Network Intrusion Detection_CIC-IDS-2017/Tuesday-WorkingHours.pcap_ISCX.csv')
data3 = pd.read_csv('../../../Dataset/Network Intrusion Detection_CIC-IDS-2017/Wednesday-workingHours.pcap_ISCX.csv')
data4 = pd.read_csv('../../../Dataset/Network Intrusion Detection_CIC-IDS-2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
data5 = pd.read_csv(
    '../../../Dataset/Network Intrusion Detection_CIC-IDS-2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
data6 = pd.read_csv('../../../Dataset/Network Intrusion Detection_CIC-IDS-2017/Friday-WorkingHours-Morning.pcap_ISCX.csv')
data7 = pd.read_csv('../../../Dataset/Network Intrusion Detection_CIC-IDS-2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
data8 = pd.read_csv('../../../Dataset/Network Intrusion Detection_CIC-IDS-2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

data_list = [data1, data2, data3, data4, data5, data6, data7, data8]

print('Data dimensions: ')
for i, data in enumerate(data_list, start=1):
    rows, cols = data.shape
    print(f'Data{i} -> {rows} rows, {cols} columns')

data = pd.concat(data_list)
rows, cols = data.shape

print('New dimension:')
print(f'Number of rows: {rows}')
print(f'Number of columns: {cols}')
print(f'Total cells: {rows * cols}')

# Deleting dataframes after concating to save memory
for d in data_list: del d

# Renaming the columns by removing leading/trailing whitespace
col_names = {col: col.strip() for col in data.columns}
data.rename(columns=col_names, inplace=True)

print(data.columns)
print(data.info())

# ********* Data Cleaning *********
#Identifying duplicate values
dups = data[data.duplicated()]
print(f'Number of duplicates: {len(dups)}')
data.drop_duplicates(inplace = True)
print(data.shape)

#Identifying missing values
missing_val = data.isna().sum()
print(missing_val.loc[missing_val>0])

# Checking for infinity values
numeric_cols = data.select_dtypes(include = np.number).columns
inf_count = np.isinf(data[numeric_cols]).sum()
print(inf_count[inf_count > 0])

# Replacing any infinite values (positive or negative) with NaN (not a number)
print(f'Initial missing values: {data.isna().sum().sum()}')
data.replace([np.inf, -np.inf], np.nan, inplace = True)
print(f'Missing values after processing infinite values: {data.isna().sum().sum()}')
missing = data.isna().sum()
print(missing.loc[missing > 0])

# Calculating missing value percentage in the dataset
mis_per = (missing / len(data)) * 100
mis_table = pd.concat([missing, mis_per.round(2)], axis = 1)
mis_table = mis_table.rename(columns = {0 : 'Missing Values', 1 : 'Percentage of Total Values'})

print(mis_table.loc[mis_per > 0])

print(data['Label'].unique())
# Types of attacks & normal instances (BENIGN)
print(data['Label'].value_counts())
# Creating a dictionary that maps each label to its attack type
attack_map = {
    'BENIGN': 'BENIGN',
    'DDoS': 'DDoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'PortScan': 'Port Scan',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Bot',
    'Web Attack � Brute Force': 'Web Attack',
    'Web Attack � XSS': 'Web Attack',
    'Web Attack � Sql Injection': 'Web Attack',
    'Infiltration': 'Infiltration',
    'Heartbleed': 'Heartbleed'
}
# Creating a new column 'Attack Type' in the DataFrame based on the attack_map dictionary
data['Attack Type'] = data['Label'].map(attack_map)
print(data['Attack Type'].value_counts())

#The Label column is permanently removed from the data DataFrame.
data.drop('Label', axis = 1, inplace = True)

le = LabelEncoder()
data['Attack Number'] = le.fit_transform(data['Attack Type'])

print(data['Attack Number'].unique())
# Printing corresponding attack type for each encoded value
encoded_values = data['Attack Number'].unique()
for val in sorted(encoded_values):
    print(f"{val}: {le.inverse_transform([val])[0]}")

# ********* Data Preprocessing *********
# For improving performance and reduce memory-related errors
old_memory_usage = data.memory_usage().sum() / 1024 ** 2
print(f'Initial memory usage: {old_memory_usage:.2f} MB')
for col in data.columns:
    col_type = data[col].dtype
    if col_type != object:
        c_min = data[col].min()
        c_max = data[col].max()
        # Downcasting float64 to float32
        if str(col_type).find('float') >= 0 and c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            data[col] = data[col].astype(np.float32)

        # Downcasting int64 to int32
        elif str(col_type).find('int') >= 0 and c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            data[col] = data[col].astype(np.int32)

new_memory_usage = data.memory_usage().sum() / 1024 ** 2
print(f"Final memory usage: {new_memory_usage:.2f} MB")

# Calculating percentage reduction in memory usage
print(f'Reduced memory usage: {1 - (new_memory_usage / old_memory_usage):.2%}')

# Dropping columns with only one unique value
num_unique = data.nunique()
one_variable = num_unique[num_unique == 1]
not_one_variable = num_unique[num_unique > 1].index

dropped_cols = one_variable.index
data = data[not_one_variable]

# ********* Incremental PCA *********
data.dropna(inplace=True)
# Standardizing the dataset
features = data.drop('Attack Type', axis = 1)
attacks = data['Attack Type']

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

size = len(features.columns) // 2
ipca = IncrementalPCA(n_components = size, batch_size = 500)
for batch in np.array_split(scaled_features, len(features) // 500):
    ipca.partial_fit(batch)

print(f'information retained: {sum(ipca.explained_variance_ratio_):.2%}')

transformed_features = ipca.transform(scaled_features)
new_data = pd.DataFrame(transformed_features, columns = [f'PC{i+1}' for i in range(size)])
new_data['Attack Type'] = attacks.values
print(new_data)
