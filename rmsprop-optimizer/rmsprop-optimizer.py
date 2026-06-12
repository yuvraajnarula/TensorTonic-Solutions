import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    w,g,s = np.array(w), np.array(g), np.array(s)
    
    snext = beta * s + (1  - beta ) * (g ** 2)
    wnext = w - ((lr / (np.sqrt(snext+ eps))) * g)
    return wnext,snext