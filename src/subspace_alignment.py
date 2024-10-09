import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

class SubspaceAlignment:
    def __init__(self, n_components=20):
        """
        Initialize the Subspace Alignment method.
        Args:
        - n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components

    def fit_transform(self, X_source, X_target):
        """
        Fit and transform the source and target domains using Subspace Alignment.
        Args:
        - X_source (np.array): Source domain feature matrix.
        - X_target (np.array): Target domain feature matrix.
        
        Returns:
        - X_source_aligned (np.array): Aligned source domain feature matrix.
        - X_target_pca (np.array): Target domain feature matrix after PCA.
        """
        # Apply PCA to both source and target domains
        pca_source = PCA(n_components=self.n_components)
        pca_target = PCA(n_components=self.n_components)
        
        # Project source and target data onto PCA subspaces
        X_source_pca = pca_source.fit_transform(X_source)
        X_target_pca = pca_target.fit_transform(X_target)
        
        # Compute the alignment matrix
        M = np.dot(pca_source.components_.T, pca_target.components_)
        
        # Align the source domain using the alignment matrix
        X_source_aligned = np.dot(X_source_pca, M)
        
        print("Subspace Alignment completed.")
        return X_source_aligned, X_target_pca

# Test the module with CaffeNet features
if __name__ == "__main__":
    from data_loader import load_data, normalize_features
    
    # Load CaffeNet
    X_webcam, y_webcam, X_dslr, y_dslr = load_data('caffenet')
    
    # Normalize features
    X_webcam_norm = normalize_features(X_webcam)
    X_dslr_norm = normalize_features(X_dslr)
    
    # Initialize and apply Subspace Alignment
    sa = SubspaceAlignment(n_components=20)
    X_webcam_aligned, X_dslr_pca = sa.fit_transform(X_webcam_norm, X_dslr_norm)
