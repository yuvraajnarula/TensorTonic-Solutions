import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0 
    for _ in range(steps):
        z = X @ w + b 
        p = _sigmoid(z)
        error_vector = p - y 
        dw = (1 / n_samples) * (X.T @ error_vector)
        db =  (1/n_samples) * np.sum(error_vector)
        w-=lr*dw 
        b-=lr*db 
    return w,b
    