import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    w,m,v,grad = np.array(w, dtype=float) , np.array(m, dtype=float), np.array(v, dtype=float), np.array(grad, dtype=float)
    m_ = beta1 * m + (1-beta1) * grad 
    v_ = beta2 * v + (1 - beta2) * (grad ** 2)
    nn = beta1 * m_ + (1 - beta1) * grad 
    dd = np.sqrt(v_)+eps
    w_ = w - lr  * (nn / dd)
    return w_, m_, v_