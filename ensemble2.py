import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load the dataset
try:
    df = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("Error: 'dataset.csv' not found. Please ensure the dataset file is in the correct directory.")
    exit()

# --- Identify Features and Target ---
# Updated column names based on the content of dataset.csv
feature_columns = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
    'CALC', 'MTRANS' # These are the 17 parameters as 'id' is an identifier
]
target_column = 'NObeyesdad' # This is the target column name from dataset.csv

# Check if all identified columns exist in the DataFrame
missing_columns = [col for col in feature_columns + [target_column] if col not in df.columns]
if missing_columns:
    print(f"Error: The following columns are missing from the dataset: {missing_columns}")
    print("Please check your feature_columns and target_column definitions.")
    exit()

X = df[feature_columns]
y = df[target_column]

from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    df = df.copy()
    
    # Encode all object-type columns (categorical) including the target
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    # Features and target
    X = df.drop('NObeyesdad', axis=1)  # Make sure this is the actual target column
    y = df['NObeyesdad']
    return X, y

# --- Initialize base models ---
def get_base_models():
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    return [('rf', rf), ('knn', knn)]

# --- Evaluation Metrics ---
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n--- {model_name.upper()} METRICS ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision (macro):", precision_score(y_true, y_pred, average='macro'))
    print("Recall (macro):", recall_score(y_true, y_pred, average='macro'))
    print("F1 Score (macro):", f1_score(y_true, y_pred, average='macro'))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# --- Cross-validated Stacking Function ---
def stacked_cv_model(X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    base_models = get_base_models()

    oof_preds = {name: np.zeros((X.shape[0], len(np.unique(y)))) for name, _ in base_models}
    test_preds = {name: [] for name, _ in base_models}
    meta_features = np.zeros((X.shape[0], len(base_models) * len(np.unique(y))))

    X_np = X.values
    y_np = y.values
   

    for name, model in base_models:
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
             X_train, y_train = X_np[train_idx], y_np[train_idx]
             X_valid, y_valid = X_np[val_idx], y_np[val_idx]
             model.fit(X_train, y_train)
             oof_pred = model.predict_proba(X_valid)
             oof_preds[name][val_idx] = oof_pred
             print(f"{name.upper()} Fold {fold + 1} complete.")

        meta_features[:, len(np.unique(y)) * list(test_preds.keys()).index(name):
                         len(np.unique(y)) * (list(test_preds.keys()).index(name) + 1)] = oof_preds[name]

        # Evaluation of base model
        y_pred_base = np.argmax(oof_preds[name], axis=1)
        evaluate_model(y, y_pred_base, model_name=name)

    # --- Train meta-model on stacked features ---
    meta_model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    meta_model.fit(meta_features, y)

    # --- Predict using meta-model ---
    final_pred = meta_model.predict(meta_features)

    # --- Final Stacked Model Evaluation ---
    evaluate_model(y, final_pred, model_name='Stacked Model')

    return meta_model

# --- Main Execution ---
# Example:
# df = pd.read_csv('ObesityDataSet.csv')
# X, y = preprocess(df)
# model = stacked_cv_model(X, y)


if __name__ == '__main__':
    df = pd.read_csv('dataset.csv')  # replace with actual filename if different
    X, y = preprocess(df)
    model = stacked_cv_model(X, y)

