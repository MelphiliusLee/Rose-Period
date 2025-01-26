import pandas as pd
import numpy as np
file_path = 'FedCycleData071012 (2) (1).csv'

# Specify the required features and label
features = ["ClientID",
    "Age", "Height", "Weight",
    "MeanBleedingIntensity", "NumberofDaysofIntercourse", "Yearsmarried",
    "MeanCycleLength"
]
label_a = "LengthofCycle"
label_b = "LengthofMenses"
label_c = "UnusualBleeding"


columns_to_load= features + [label_a] +[label_b] + [label_c]
df_origin = pd.read_csv(file_path, usecols=columns_to_load)
df_filtered = df_origin.drop_duplicates(subset="ClientID", keep="first")
df_filtered= df_filtered.replace('', np.nan)
df_filtered = df_filtered.apply(pd.to_numeric, errors='coerce')
df_filtered= df_filtered.fillna(df_filtered.mean())

df_filtered= df_filtered.astype(float)
print(df_filtered.dtypes)
df = df_filtered.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['float64', 'int64'] else col)
X = df[features]
y_a = df[label_a]
y_b = df[label_b]
y_c = df[label_c]

print(df.head())

print("Features (X):")
print(X.head())

print("\nLabel (y):")
print(y_a.head())
print(y_b.head())
print(y_c.head())