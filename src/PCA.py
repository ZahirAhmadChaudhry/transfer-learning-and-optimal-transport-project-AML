
import numpy as np
from sklearn.decomposition import PCA

class PCAHandler:
    def __init__(self, n_components=20):
        """
        Initialize the PCA handler.
        Args:
        - n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components
        self.pca = None

    def fit(self, X):
        """
        Fit PCA on the provided feature matrix.
        Args:
        - X (np.array): Feature matrix of shape (n_samples, n_features).
        
        Returns:
        - X_pca (np.array): PCA-transformed feature matrix.
        """
        self.pca = PCA(n_components=self.n_components)
        X_pca = self.pca.fit_transform(X)
        print(f"PCA fitted with {self.n_components} components. Shape after PCA: {X_pca.shape}")
        return X_pca

    def transform(self, X):
        """
        Transform the provided feature matrix using the fitted PCA.
        Args:
        - X (np.array): Feature matrix of shape (n_samples, n_features).
        
        Returns:
        - X_pca (np.array): PCA-transformed feature matrix.
        """
        if self.pca is None:
            raise ValueError("PCA model has not been fitted yet. Call `fit` first.")
        X_pca = self.pca.transform(X)
        return X_pca

    def get_components(self):
        """
        Retrieve the PCA components after fitting.
        Returns:
        - components (np.array): Matrix of PCA components.
        """
        if self.pca is None:
            raise ValueError("PCA model has not been fitted yet. Call `fit` first.")
        return self.pca.components_

