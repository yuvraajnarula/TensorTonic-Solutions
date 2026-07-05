import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    X = np.array(x)
    if p == 0.0:
        return X, np.ones_like(X)
    if rng is not None:
        random_values = rng.random(X.shape)
    else:
        Oed = np.random.random(X.shape)

    dropout_pattern = (random_values < (1-p)) / (1 - p)
    output = X * dropout_pattern
    return output, dropout_pattern