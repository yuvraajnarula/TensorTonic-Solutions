import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.array(X)
    X_mean = np.mean(X, axis=0)
    X_C = X - X_mean 
    n = X.shape[0]
    cov = (1 / (n - 1) ) * (X_C.T @ X_C)
    eigen_vals, eigen_vec = np.linalg.eigh(cov)
    idx=  np.argsort(eigen_vals)[::-1]
    eigen_vec = eigen_vec[:, idx]
    W = eigen_vec[:, :k]
    X_proj = X_C@W 
    return X_proj