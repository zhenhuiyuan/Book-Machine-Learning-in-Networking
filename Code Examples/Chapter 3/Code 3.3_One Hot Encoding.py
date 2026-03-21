#Copyright@Zhenhui Yuan, 2025
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Sample dataset
data = {
    'Protocol': ['TCP', 'UDP', 'ICMP', 'TCP', 'UDP'],
    'Latency (ms)': [20, 35, 15, 50, 25]
}
df = pd.DataFrame(data)

# Label Encoding
label_encoder = LabelEncoder()
df['Protocol_Label'] = label_encoder.fit_transform(df['Protocol'])

# One-Hot Encoding
one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = one_hot_encoder.fit_transform(df[['Protocol']])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(['Protocol']))

# Combine with original dataframe
df = pd.concat([df, one_hot_df], axis=1)

pd.set_option('display.max_columns', None) #display all columns
print("Original Data:")
print(df)
