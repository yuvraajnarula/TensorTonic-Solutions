import numpy as np

def manhattan_distance(x, y):
    x,y = np.array(x), np.array(y)
    dist = np.sum(abs(x-y))
    return int(dist)