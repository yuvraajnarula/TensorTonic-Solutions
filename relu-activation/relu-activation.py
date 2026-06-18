import numpy as np

def relu(x):
    x = np.array(x)
    x = np.maximum(0,x)

    return x