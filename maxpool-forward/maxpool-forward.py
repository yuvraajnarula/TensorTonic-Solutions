import numpy as np
def maxpool_forward(X, pool_size, stride):
    X =  np.array(X)
    rows, cols = X.shape
    
    out_rows = (rows - pool_size) // stride + 1
    out_cols = (cols - pool_size) // stride + 1
    
    res = np.zeros((out_rows, out_cols))
    
    for i in range(out_rows):
        for j in range(out_cols):
            row_start = i * stride
            row_end = row_start + pool_size
            col_start = j * stride
            col_end = col_start + pool_size
            
            window = X[row_start:row_end, col_start:col_end]
            res[i, j] = np.max(window)
    return res.tolist()