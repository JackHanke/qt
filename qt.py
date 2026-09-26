from time import time
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
# from einops import rearrange, repeat

# from flashattn.flash_attn import MHA
# from flash_attn import MHA
# from flash_attn.modules.mha import MHA

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def generate_alibi_bias(H: int):
    """Returns an alibi bias score_mod given the number of heads H
    """

    def alibi_mod(score, b, h, q_idx, kv_idx):
        scale = torch.exp2(-((h + 1) * 8.0 / H))
        bias = (kv_idx - q_idx) * scale
        return score + bias

    return alibi_mod

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

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
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_heads_kv: int,
            seq_len: int,
            use_alibi: bool,
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.seq_len = seq_len

        self.head_dim = embed_dim // num_heads
        self.qkv_dim = self.head_dim * (num_heads + 2 * num_heads_kv)
        self.total_heads = num_heads + 2 * num_heads_kv

        self.Wqkv = torch.nn.Linear(embed_dim, self.qkv_dim, bias=True)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)

        self.block_mask = create_block_mask(
            causal,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
        ).to(torch.device('cpu'))

        if use_alibi:
            self.alibi = generate_alibi_bias(self.num_heads)
        else:
            self.alibi = None

    def forward(self, x):
        B = x.shape[0]
        S = x.shape[1]

        qkv = self.Wqkv(x)
        qkv = qkv.view(B, S, self.total_heads, self.head_dim)

        query = qkv[:, :, :self.num_heads, :]
        key   = qkv[:, :, self.num_heads : self.num_heads + self.num_heads_kv, :]
        value = qkv[:, :, self.num_heads + self.num_heads_kv :, :]

        query = query.transpose(1, 2)
        key   = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # self.block_mask = self.block_mask._adjust(S,S) # NOTE this does not work on CPU

        x = flex_attention(
            query,
            key,
            value,
            score_mod=self.alibi,
            block_mask=self.block_mask,
            enable_gqa=True,
        )

        x = x.transpose(1,2)
        x = x.view(B, S, -1)
        x = self.out_proj(x)

        return x


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
                dtype=None,
            ), # NoPE every 4
            RMSNorm(d_model),
            FeedForward(dim = d_model),
        ]) for layer_idx in range(n_layers)])

        self.norm = RMSNorm(d_model)

    def forward(self, x, do_viz: bool = False):
        x = self.embeddings(x)

        if do_viz: embeds = [x.detach().cpu()]

        for i, (norm1, attn, norm2, ff) in enumerate(self.layers):
            attn_out = attn(norm1(x))
            x = x + attn_out
            if do_viz: embeds.append(x.detach().cpu())
            x = x + ff(norm2(x))
            if do_viz: embeds.append(x.detach().cpu())

        logits = self.output_linear(self.norm(x)).transpose(1,2)
        if do_viz: return logits, embeds
        return logits
    
    # @torch.no_grad()
    # def top_p(self, p)


    # @torch.no_grad()
    # def generate(self, context:str):
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


class qtflex(nn.Module):
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
            qtAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                num_heads_kv=n_heads_kv,
                use_alibi=(layer_idx % 4 == 3),
                seq_len=self.max_seq_len
            ), # NoPE every 4
            RMSNorm(d_model),
            FeedForward(dim = d_model),
        ]) for layer_idx in range(n_layers)])

        self.norm = RMSNorm(d_model)

    def forward(self, x, do_viz: bool = False):
        x = self.embeddings(x)

        # print(f'x shape: {x.shape}')
        if do_viz: embeds = [x.detach().cpu()]

        for i, (norm1, attn, norm2, ff) in enumerate(self.layers):
            attn_out = attn(norm1(x))
            x = x + attn_out
            if do_viz: embeds.append(x.detach().cpu())
            x = x + ff(norm2(x))
            if do_viz: embeds.append(x.detach().cpu())

        logits = self.output_linear(self.norm(x)).transpose(1,2)
        if do_viz: return logits, embeds
        return logits
