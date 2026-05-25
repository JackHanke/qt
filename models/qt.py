from time import time
import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

def exists(v): return v is not None

def flash_attn_with_pope(
    q,
    k,
    v,
    pos_emb = None,
    mask = None,
    causal = False,
    softmax_scale = None,
    fused = None,
    head_dimension_at_first = True,
    dropout = 0.
):
    seq_dim = 2 if head_dimension_at_first else 1
    q_len, kv_len, device = q.shape[seq_dim], k.shape[seq_dim], q.device

    # non-fused manual path
    # standardize to (batch, heads, seq, dim)

    if not head_dimension_at_first:
        q = rearrange(q, 'b n h d -> b h n d')
        k = rearrange(k, 'b n h d -> b h n d')
        v = rearrange(v, 'b n h d -> b h n d')

    q, k = apply_pope_to_qk(pos_emb, q, k, to_magnitude = torch.nn.functional.softplus)

    # group query attention support
    groups = q.shape[1] // k.shape[1]
    k = repeat(k, 'b h ... -> b (g h) ...', g = groups)
    v = repeat(v, 'b h ... -> b (g h) ...', g = groups)

    # manual attention path using SDPA
    # ensure dtypes match for SDPA (apply_pope_to_qk might have upcasted to float32)

    v_dtype = v.dtype
    v_dim = v.shape[-1]

    if q.dtype != v.dtype:
        v = v.to(q.dtype)

    attn_mask = None
    if exists(mask):
        attn_mask = rearrange(mask, 'b j -> b 1 1 j')

    if causal and q_len < kv_len:
        causal_mask = torch.ones((q_len, kv_len), dtype = torch.bool, device = device).tril(diagonal = kv_len - q_len)
        attn_mask = and_masks(attn_mask, causal_mask)
        causal = False

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask = attn_mask,
        is_causal = causal,
        scale = softmax_scale,
        dropout_p = dropout
    )

    # mps sdpa bug (pytorch 2.9.1) - output takes q/k dim instead of v dim
    # first v_dim elements are correct, so slicing suffices
    # only triggers in no_grad (inference). todo - remove once fixed upstream

    if out.shape[-1] != v_dim:
        out = out[..., :v_dim]

    out = out.to(v_dtype)

    if not head_dimension_at_first:
        out = rearrange(out, 'b h n d -> b n h d')

    return out

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

class CausalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, use_pope: bool = True):
        super().__init__()
        self.heads = heads
        self.use_pope = use_pope
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Linear(dim, dim, bias = False)

    def forward(self, x, pos_emb = None, cache = None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if exists(cache):
            ck, cv = cache
            k, v = (torch.cat(t, dim = -2) for t in ((ck, k), (cv, v)))

        new_cache = (k, v)

        if self.use_pope and exists(pos_emb):
            out = flash_attn_with_pope(
                q, k, v,
                pos_emb = pos_emb,
                causal = True,
                softmax_scale = self.scale,
                fused = True,
                head_dimension_at_first = True
            )
        else:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal = True, scale = self.scale)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), new_cache

class qt(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        seq_len: int,
        num_embeddings: int,
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
            CausalAttention(d_model, heads = n_heads, use_pope = bool((layer_idx+1) % 2)), # NoPE alternating pos encoding
            RMSNorm(d_model),
            FeedForward(dim = d_model),
        ]) for layer_idx in range(n_layers)])

        self.norm = RMSNorm(d_model)

    def forward(self, x, cache = None, return_cache = False):
        seq_len = x.shape[1]
        x = self.token_emb(x)

        seq_len_kv = seq_len if not exists(cache) else (cache[0][0].shape[-2] + seq_len)

        new_caches = []
        for i, (norm1, attn, norm2, ff) in enumerate(self.layers):
            if ((i+1)%2) == 0:
                pos_emb = self.pope(seq_len_kv)
            else:
                pos_emb - None

            layer_cache = cache[i] if exists(cache) else None
            attn_out, new_layer_cache = attn(norm1(x), pos_emb, cache = layer_cache)
            new_caches.append(new_layer_cache)
            x = x + attn_out
            x = x + ff(norm2(x))

        logits = self.output_linear(self.norm(x))

        if return_cache:
            return logits, new_caches

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