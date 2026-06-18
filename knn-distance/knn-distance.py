import numpy as np

def knn_distance(X_train, X_test, k):
    X_train,X_test = np.array(X_train), np.array(X_test)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1,1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train,n_test = X_train.shape[0], X_test.shape[0]
    d = np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=-1) ** 0.5 
    sorted_indices = np.argsort(d, axis=1)
    if k <= n_train:
        res = sorted_indices[:, :k]
    else:
        res = result = np.full((n_test, k), -1, dtype=int)
        res[:, :n_train] = sorted_indices
    return res