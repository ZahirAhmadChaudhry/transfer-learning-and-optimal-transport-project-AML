# domain_adaptation.py

import numpy as np
from sklearn.decomposition import PCA
import ot

def subspace_alignment(S, T, d):
    """
    Perform Subspace Alignment domain adaptation.

    Args:
        S (np.ndarray): Source data matrix.
        T (np.ndarray): Target data matrix.
        d (int): Number of principal components.

    Returns:
        S_aligned (np.ndarray): Aligned source data.
        T_pca (np.ndarray): Projected target data.
    """
    # Step 1: PCA on source and target
    pca_source = PCA(n_components=d)
    S_pca = pca_source.fit_transform(S)
    Xs = pca_source.components_

    pca_target = PCA(n_components=d)
    T_pca = pca_target.fit_transform(T)
    Xt = pca_target.components_

    # Step 2: Compute alignment matrix
    M = np.dot(Xs, Xt.T)

    # Step 3: Align source data
    S_aligned = np.dot(S_pca, M)

    return S_aligned, T_pca

def optimal_transport(S, T, lambda_reg):
    """
    Perform Optimal Transport domain adaptation.

    Args:
        S (np.ndarray): Source data matrix.
        T (np.ndarray): Target data matrix.
        lambda_reg (float): Entropic regularization parameter.

    Returns:
        S_aligned (np.ndarray): Transported source data.
        T (np.ndarray): Target data (unchanged).
    """
    n_S = S.shape[0]
    n_T = T.shape[0]
    a = np.ones((n_S,)) / n_S  # Uniform distribution over source samples
    b = np.ones((n_T,)) / n_T  # Uniform distribution over target samples

    # Compute cost matrix (Euclidean distance)
    M = ot.dist(S, T, metric='euclidean')
    M /= M.max()  # Normalize

    # Compute coupling matrix
    gamma = ot.sinkhorn(a, b, M, lambda_reg)

    # Transport source samples
    S_aligned = np.dot(gamma, T)

    return S_aligned, T
