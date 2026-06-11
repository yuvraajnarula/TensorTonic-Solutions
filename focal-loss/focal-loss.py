import numpy as np

def focal_loss(p, y, gamma=2.0):
    p_, y_ = np.array(p), np.array(y)
    clip_p = np.clip(p_, 1e-15, 1.0 - (1e-15))
    pos_term = ((1 - clip_p) ** gamma) * y_ * np.log(clip_p)
    neg_term = (clip_p ** gamma) * (1 - y_) * np.log(1 -clip_p)
    
    loss = -np.mean(pos_term + neg_term)
    
    return loss