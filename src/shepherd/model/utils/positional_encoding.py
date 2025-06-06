import math
import torch

def positional_encoding(position, dim, device):
    if len(position.shape) == 1:
        position = position[None,:].T
    assert len(position.shape) == 2
    assert position.shape[1] == 1
    assert dim % 2 == 0
    
    # position has shape (B, 1)
    # dim is scalar
    # returns position embeddings of shape (B, dim)
    
    pe = torch.zeros(position.shape[0], dim, device = device)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float, device = device) * -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe