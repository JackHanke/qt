import math
import torch
import torch.nn as nn
from time import time
from dataclasses import dataclass

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim = -1) * self.scale * self.gamma

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class qtAttention(nn.Module):
    def __init__():
        # TODO RoPE or PoPE, use FlexAttention, XSA, 
        pass

@dataclass
class qtConfig:
    '''
    numerical configurations for qt arch
    '''
    D_MODEL = 2048
    N_LAYERS = 22
    N_HEADS = 32
    N_HEADS_KV = 8 
    NUM_EMBEDDINGS = 10_001
    INIT_MEAN = 0.0
    INIT_STD = 0.02

class qt2(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_heads_kv: int,
        seq_len: int,
        num_embeddings: int,
        device,
    ):
        '''
        the qt2 arch
        '''

        super().__init__()
        self.max_seq_len = seq_len

        # embeddings, tied!
        self.embeddings = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=d_model,
        )
        self.output_linear = nn.Linear(d_model, num_embeddings, bias=False)
        self.output_linear.weight = self.embeddings.weight

        self.layers = nn.ModuleList([nn.ModuleList([
            RMSNorm(d_model),
            qtAttention(),
            RMSNorm(d_model),
            FeedForward(dim = d_model),
        ]) for layer_idx in range(n_layers)])

        self.norm = RMSNorm(d_model)

        
        

    def forward(self, x, do_viz: bool = False):
        x = self.embeddings(x)
        if do_viz: embeds = [x.detach().cpu()]

        for i, (norm1, attn, norm2, ff) in enumerate(self.layers):
            attn_out = norm1(attn(x))
            x = x + attn_out
            if do_viz: embeds.append(x.detach().cpu())
            x = x + norm2(ff(x))
            if do_viz: embeds.append(x.detach().cpu())

        logits = self.output_linear(self.norm(x)).transpose(1,2)
        if do_viz: return logits, embeds
        return logits
    