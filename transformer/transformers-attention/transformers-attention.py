import torch
import torch.nn.functional as F
import math
    
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    K_T = torch.transpose(K, -2, -1)
    _  = torch.matmul(Q, K_T) / (K.shape[-1] ** 0.5)
    res = torch.matmul(F.softmax(_, dim=-1), V)
    return res