# preprocessing.py

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest

def standardize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def handle_class_imbalance(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def remove_outliers(X, y, contamination=0.05):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    y_pred = iso_forest.fit_predict(X)
    mask = y_pred != -1
    X_clean = X[mask]
    y_clean = y[mask]
    return X_clean, y_clean

def preprocess_data(X, y, standardize=True, balance_classes=True, remove_outlier=True):
    if standardize:
        X = standardize_data(X)
    if balance_classes:
        X, y = handle_class_imbalance(X, y)
    if remove_outlier:
        X, y = remove_outliers(X, y)
    return X, y
