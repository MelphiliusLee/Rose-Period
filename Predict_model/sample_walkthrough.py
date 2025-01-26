from lango_models import LangoModels
import pandas as pd
import numpy as np
# Initialize the LangoModels class
lango = LangoModels(file_path='FedCycleData071012 (2) (1).csv')

# Train and evaluate all models
lango.train_and_evaluate()

# Predict with model_a (numerical prediction)


sample_input = lango.X_test_n.iloc[:1]
print(sample_input)
print("\nPredictions for y_a (Length of Cycle):", lango.predict("model_LengthofCycle", sample_input))

# Predict with model_c (binary classification)
sample_input = lango.X_test_c.iloc[:1]
print(sample_input)
print("\nPredictions for y_c (Unusual Bleeding):", lango.predict("model_UnusualBleeding", sample_input))

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
df = df_filtered.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['float64', 'int64'] else col)

X = df[features]
X=X.drop(columns=["ClientID"])
print("mean:", X["MeanCycleLength"].mean())
sample_input = X.iloc[:1]
print(sample_input)
print("\nPredictions for y_a (Length of Cycle):", lango.predict("model_UnusualBleeding", sample_input))


