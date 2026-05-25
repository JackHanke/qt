from time import time
import math
import torch
import torch.nn as nn
# from einops import rearrange, repeat

# from flashattn.flash_attn import MHA
# from flash_attn import MHA
from flash_attn.modules.mha import MHA

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

class qt(nn.Module):
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
            MHA(
                embed_dim=d_model,
                num_heads=n_heads,
                num_heads_kv=n_heads_kv,
                causal=True,
                use_alibi=(layer_idx % 4 == 3),
                fused_bias_fc=False,
                use_flash_attn=True,
                device=device,
                dtype=torch.bfloat16,
            ), # NoPE every 4
            RMSNorm(d_model),
            FeedForward(dim = d_model),
        ]) for layer_idx in range(n_layers)])

        self.norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.embeddings(x)

        for i, (norm1, attn, norm2, ff) in enumerate(self.layers):
            attn_out = attn(norm1(x))
            x = x + attn_out
            x = x + ff(norm2(x))

        logits = self.output_linear(self.norm(x)).transpose(1,2)
        return logits

    # @torch.no_grad()
    # def generate(self, prompts, seq_len, temperature = 1.0, filter_thres = 0.9):
    #     b, t = prompts.shape
    #     out = prompts
    #     cache = None

    #     for _ in tqdm.tqdm(range(seq_len), desc='generating'):
    #         curr_x = out[:, -self.max_seq_len:] if not exists(cache) else out[:, -1:]
    #         logits, cache = self.forward(curr_x, cache = cache, return_cache = True)
    #         logits = logits[:, -1]

    #         # top-k filtering
    #         logits = top_k(logits, thres = filter_thres)

    #         probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
    #         sample = torch.multinomial(probs, 1)
    #         out = torch.cat((out, sample), dim=-1)
    #     return out[:, t:]
