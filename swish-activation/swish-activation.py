import numpy as np

def swish(x):
    x = np.array(x)
    return x / (1.0 + np.exp(-1 * x))