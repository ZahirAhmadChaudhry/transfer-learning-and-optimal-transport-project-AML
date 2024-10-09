import scipy.io
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(feature_type='surf'):
    """
    Load SURF or CaffeNet features for both Webcam and DSLR domains.
    
    Args:
    - feature_type (str): 'surf' or 'CaffeNet' to specify which feature type to load.
    
    Returns:
    - X_webcam (np.array): Feature matrix for Webcam domain.
    - y_webcam (np.array): Labels for Webcam domain.
    - X_dslr (np.array): Feature matrix for DSLR domain.
    - y_dslr (np.array): Labels for DSLR domain.
    """
    base_dir = 'data/'  # Base directory for data

    # Directories based on feature type
    if feature_type.lower() == 'surf':
        data_dir = os.path.join(base_dir, 'surf/')
    elif feature_type.lower() == 'caffenet':
        data_dir = os.path.join(base_dir, 'CaffeNet/')
    else:
        raise ValueError("Invalid feature type specified. Use 'surf' or 'CaffeNet'.")

    # File paths for the Webcam and DSLR domains
    webcam_path = os.path.join(data_dir, 'webcam.mat')
    dslr_path = os.path.join(data_dir, 'dslr.mat')

    # Load data from .mat files
    data_webcam = scipy.io.loadmat(webcam_path)
    X_webcam = data_webcam['fts']   # Feature matrix
    y_webcam = data_webcam['labels'].flatten()  # Labels

    data_dslr = scipy.io.loadmat(dslr_path)
    X_dslr = data_dslr['fts']  # Feature matrix
    y_dslr = data_dslr['labels'].flatten()  # Labels

    print(f"Loaded {feature_type} features:")
    print(f" - Webcam: X shape = {X_webcam.shape}, y shape = {y_webcam.shape}")
    print(f" - DSLR: X shape = {X_dslr.shape}, y shape = {y_dslr.shape}")

    return X_webcam, y_webcam, X_dslr, y_dslr

def normalize_features(X):
    """
    Normalize the features using z-score normalization.
    Args:
    - X (np.array): Feature matrix.
    
    Returns:
    - X_normalized (np.array): Normalized feature matrix.
    """
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized

if __name__ == "__main__":
    # Load and normalize Caffenet features for testing
    X_webcam, y_webcam, X_dslr, y_dslr = load_data('CaffeNet')
    X_webcam_norm = normalize_features(X_webcam)
    X_dslr_norm = normalize_features(X_dslr)