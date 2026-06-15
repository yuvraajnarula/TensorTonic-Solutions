import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if not seqs :
        return np.empty((0,0), dtype=int)

    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    n = len(seqs)
    p = np.full((n, max_len), pad_value, dtype=int)
    for i,seq in enumerate(seqs):
        trunc = seq[:max_len]
        p[i, :len(trunc)] = trunc 
    return p