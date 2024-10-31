# data_loading.py

import os
from scipy.io import loadmat
import numpy as np

def load_dataset(domain, feature_type, data_dir='../data'):
    """
    Load the specified dataset and feature type.

    Args:
        domain (str): The domain to load ('webcam', 'dslr', 'amazon', 'caltech').
        feature_type (str): The feature type ('surf', 'caffenet').
        data_dir (str): The directory where data is stored.

    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
    """
    domain = domain.lower()
    feature_type = feature_type.lower()

    if feature_type == 'surf':
        feature_dir = os.path.join(data_dir, 'Surf')
    elif feature_type == 'caffenet':
        feature_dir = os.path.join(data_dir, 'CaffeNet')
    else:
        raise ValueError("Invalid feature type. Choose 'surf' or 'caffenet'.")

    file_path = os.path.join(feature_dir, f'{domain}.mat')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = loadmat(file_path)
    X = data['fts']
    y = data['labels'].flatten()

    return X, y
