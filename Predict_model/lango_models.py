import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix
from imblearn.over_sampling import SMOTE

class LangoModels:
    def __init__(self, file_path):
        self.file_path = file_path
        self.features = [
            "ClientID", "Age", "Height", "Weight",
            "MeanBleedingIntensity", "NumberofDaysofIntercourse",
            "Yearsmarried", "MeanCycleLength"
        ]
        self.label_a = "LengthofCycle"
        self.label_b = "LengthofMenses"
        self.label_c = "UnusualBleeding"

        self.df = None
        self.model_a = None
        self.model_b = None
        self.model_c = None

        self.X_train_c, self.X_test_c = None, None
        self.y_train_c, self.y_test_c = None, None

        self.X_train_n, self.X_test_n = None, None
        self.y_train_a, self.y_test_a = None, None
        self.y_train_b, self.y_test_b = None, None

        self.data_processing()
        self.build_models()

    def data_processing(self):
        """
        Load, clean, and preprocess data.
        """
        columns_to_load = self.features + [self.label_a, self.label_b, self.label_c]
        self.df = pd.read_csv(self.file_path, usecols=columns_to_load)

        # Drop duplicates based on ClientID
        self.df = self.df.drop_duplicates(subset="ClientID", keep="first")
        self.df = self.df.replace('', np.nan).apply(pd.to_numeric, errors='coerce')
        self.df = self.df.fillna(self.df.mean())

        # Prepare data for binary classification
        X = self.df[self.features].drop(columns=["ClientID"])
        y_c = self.df[self.label_c].astype(int)

        smote = SMOTE(random_state=42)
        X_smote, y_c_smote = smote.fit_resample(X, y_c)
        self.X_train_c, self.X_test_c, self.y_train_c, self.y_test_c = train_test_split(
            X_smote, y_c_smote, test_size=0.2, random_state=23
        )

        # Prepare data for numerical prediction
        y_a = self.df[self.label_a]
        y_b = self.df[self.label_b]
        self.X_train_n, self.X_test_n, self.y_train_a, self.y_test_a = train_test_split(
            X, y_a, test_size=0.2, random_state=23
        )
        _, _, self.y_train_b, self.y_test_b = train_test_split(
            X, y_b, test_size=0.2, random_state=23
        )

    def build_models(self):
        """
        Build and initialize models.
        """
        # Binary classification model for y_c
        self.model_c = RandomForestClassifier(random_state=42)

        # Regression models for y_a and y_b
        self.model_a = RandomForestRegressor(random_state=42)
        self.model_b = RandomForestRegressor(random_state=42)

    def train_and_evaluate(self):
        """
        Train and evaluate all models.
        """
        # Train and evaluate model_c (binary classification)
        self.model_c.fit(self.X_train_c, self.y_train_c)
        y_pred_c = self.model_c.predict(self.X_test_c)
        cm = confusion_matrix(self.y_test_c, y_pred_c)
        accuracy_c = accuracy_score(self.y_test_c, y_pred_c)
        TN, FP, FN, TP = cm.ravel()
        false_negative_rate = FN / (FN + TP)
        false_positive_rate = FP / (FP + TN)

        print("\nBinary Classification Results (y_c):")
        print("Accuracy:", accuracy_c)
        print("Confusion Matrix:\n", cm)
        print("False Negative Rate:", false_negative_rate)
        print("False Positive Rate:", false_positive_rate)

        # Train and evaluate model_a (regression for y_a)
        self.model_a.fit(self.X_train_n, self.y_train_a)
        y_pred_a = self.model_a.predict(self.X_test_n)
        mae_a = mean_absolute_error(self.y_test_a, y_pred_a)
        mse_a = mean_squared_error(self.y_test_a, y_pred_a)

        print("\nNumerical Prediction Results for y_a (Length of Cycle):")
        print("Mean Absolute Error (MAE):", mae_a)
        print("Mean Squared Error (MSE):", mse_a)

        # Train and evaluate model_b (regression for y_b)
        self.model_b.fit(self.X_train_n, self.y_train_b)
        y_pred_b = self.model_b.predict(self.X_test_n)
        mae_b = mean_absolute_error(self.y_test_b, y_pred_b)
        mse_b = mean_squared_error(self.y_test_b, y_pred_b)

        print("\nNumerical Prediction Results for y_b (Length of Menses):")
        print("Mean Absolute Error (MAE):", mae_b)
        print("Mean Squared Error (MSE):", mse_b)

    def predict(self, model_type, X_input):
        """
        Predict using a specified model.
        Args:
            model_type (str): 'model_a', 'model_b', or 'model_c'.
            X_input (DataFrame): Input data for prediction.

        Returns:
            Predicted values.
        """
        if model_type == "model_a":
            return self.model_a.predict(X_input)
        elif model_type == "model_b":
            return self.model_b.predict(X_input)
        elif model_type == "model_c":
            return self.model_c.predict(X_input)
        else:
            raise ValueError("Invalid model_type. Choose 'model_a', 'model_b', or 'model_c'.")
