import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # Example model
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix
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
print(X.head())
y_a = df[label_a]
y_b = df[label_b]
y_c = df[label_c]

y_a = y_a.replace('', np.nan)
y_a = y_a.apply(pd.to_numeric, errors='coerce')
y_a = y_a.fillna(y_b.mean())
y_a= y_a.astype(float)

y_b = y_b.replace('', np.nan)
y_b = y_b.apply(pd.to_numeric, errors='coerce')
y_b = y_b.fillna(y_b.mean())
y_c = y_c.astype(float)

y_c = y_c.replace('', np.nan)
y_c = y_c.apply(pd.to_numeric, errors='coerce')
y_c = y_c.fillna(0)
y_c = y_c.astype(int)

percentage_unusual_bleeding = (y_c.sum() / len(y_c)) * 100

print(f"Percentage of UnusualBleeding = 1: {percentage_unusual_bleeding:.2f}%")

smote = SMOTE(random_state=42)
X_smote, y_c_smote = smote.fit_resample(X, y_c)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_smote, y_c_smote, test_size=0.2, random_state=23
)

print("Training Features Shape:", X_train_c.shape)
print("Testing Features Shape:", X_test_c.shape)
print("Training Labels Shape:", y_train_c.shape)
print("Testing Labels Shape:", y_test_c.shape)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_c, y_train_c)

y_pred = model.predict(X_test_c)

y_proba = model.predict_proba(X_test_c)[:, 1]

accuracy = accuracy_score(y_test_c, y_pred)
print(accuracy)
# print("a random choice")
# print(model.predict(X.iloc[20]))

cm = confusion_matrix(y_test_c, y_pred)
print("Confusion Matrix:\n", cm)


TN, FP, FN, TP = cm.ravel()

false_negative_rate = FN / (FN + TP)
false_positive_rate = FP / (FP + TN)

print("False Negative Rate (FNR):", false_negative_rate)
print("False Positive Rate (FPR):", false_positive_rate)

X_train_n, X_test_n, y_train_a, y_test_a = train_test_split(
    X, y_a, test_size=0.2, random_state=23
)

_, _, y_train_b, y_test_b = train_test_split(
    X, y_b, test_size=0.2, random_state=23
)

# Train the models for numerical prediction
model_a = RandomForestRegressor(random_state=42)
model_a.fit(X_train_n, y_train_a)

model_b = RandomForestRegressor(random_state=42)
model_b.fit(X_train_n, y_train_b)

# Make predictions for numerical targets
y_pred_a = model_a.predict(X_test_n)
y_pred_b = model_b.predict(X_test_n)

# Evaluate numerical predictions
mae_a = mean_absolute_error(y_test_a, y_pred_a)
mse_a = mean_squared_error(y_test_a, y_pred_a)

mae_b = mean_absolute_error(y_test_b, y_pred_b)
mse_b = mean_squared_error(y_test_b, y_pred_b)

print("\nNumerical Prediction Results for y_a (Length of Cycle):")
print("Mean Absolute Error (MAE):", mae_a)
print("Mean Squared Error (MSE):", mse_a)

print("\nNumerical Prediction Results for y_b (Length of Menses):")
print("Mean Absolute Error (MAE):", mae_b)
print("Mean Squared Error (MSE):", mse_b)