import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    x_mean = np.mean(X, axis=axis, keepdims=True)
    X_std = np.std(X, axis=axis, keepdims=True) + eps
    z = (X - x_mean) / X_std
    return z