import numpy as np

def leaky_relu(x, alpha=0.01):
    x= np.array(x)
    x = np.maximum(alpha * x, x)
    return x