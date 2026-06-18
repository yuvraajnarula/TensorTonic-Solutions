import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    pe = np.zeros((seq_length, d_model))

    p = np.arange(seq_length).reshape(-1, 1)
    div_term = np.exp(np.arange(0,d_model,2) * (-np.log(10000.0)/ d_model))
    div_term_indices = np.arange(0, d_model, 2)
    pe[:, 0::2] = np.sin(p * div_term)
    pe[:, 1::2] = np.cos(p * div_term[:d_model // 2])
    
    return pe