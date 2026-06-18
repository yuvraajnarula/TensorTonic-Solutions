import numpy as np

def euclidean_distance(x, y):
    x,y = np.array(x), np.array(y)
    dist = float(np.sum((x - y) ** 2)) ** 0.5
    return dist