import numpy as np

def auc(fpr, tpr):
    fpr, tpr = np.array(fpr), np.array(tpr)
    area = 0.5 * np.sum((tpr[1:] + tpr[:-1]) * (np.diff(fpr)))
    return area
