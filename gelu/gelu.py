import numpy as np
import math

def gelu(x):
    x = np.array(x)
    _ = x / ( 2 ** 0.5)
    
    vec_erf = np.vectorize(math.erf)
    _ = vec_erf(_) + 1
    x = _ * x * 0.5 
    return x
    