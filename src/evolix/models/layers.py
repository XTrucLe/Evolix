import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel
from typing import Optional, Tuple

from evolix.utils import SDPA_BACKEND
from evolix.kernels.triton_ops import fused_kv_down_rms, fused_rope, mla_flash_attention


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        max_seq_len: int = 2048,
        lora_rank: int = 128,
        rope_dim: int = 64,
    ):
        super().__init__()

        assert dim % heads == 0

        self.dim = dim
        self.heads = heads

        self.head_dim = dim // heads
        self.rope_dim = rope_dim
        self.nope_dim = self.head_dim

        self.scale = self.head_dim**-0.5

        self.q_nope = nn.Linear(dim, heads * self.nope_dim, bias=False)
        self.q_rope = nn.Linear(dim, heads * rope_dim, bias=False)

        self.kv_down = nn.Linear(dim, lora_rank, bias=False)
        self.kv_up_k = nn.Linear(lora_rank, heads * self.nope_dim, bias=False)
        self.kv_up_v = nn.Linear(lora_rank, heads * self.nope_dim, bias=False)
        self.kv_norm_w = nn.Parameter(torch.ones(lora_rank))

        self.k_rope_proj = nn.Linear(dim, rope_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_rope_cache(max_seq_len)

    def _build_rope_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)

        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

        self._cached_len = seq_len

    def _get_cos_sin(self, T, offset=0):
        needed = offset + T
        if needed > self._cached_len:
            self._build_rope_cache(max(needed, self._cached_len * 2))
        return self.cos_cache[offset : offset + T], self.sin_cache[offset : offset + T]

    def forward(self, x, offset=0, kv_cache=None):
        B, T, C = x.shape
        H = self.heads
        Dk = self.head_dim
        Dr = self.rope_dim

        q_nope = self.q_nope(x).view(B, T, H, Dk)
        q_rope = self.q_rope(x).view(B, T, H, Dr)

        kv_latent = fused_kv_down_rms(x, self.kv_down.weight, self.kv_norm_w)

        k_nope = self.kv_up_k(kv_latent).view(B, T, H, Dk)
        v = self.kv_up_v(kv_latent).view(B, T, H, Dk)

        k_rope = self.k_rope_proj(x).view(B, T, 1, Dr)

        cos, sin = self._get_cos_sin(T, offset)

        q_rope, k_rope = fused_rope(q_rope, k_rope, cos, sin)

        if kv_cache is not None and not self.training:
            lat_prev, rope_prev = kv_cache

            kv_latent = torch.cat([lat_prev, kv_latent], dim=1)
            k_rope = torch.cat([rope_prev, k_rope], dim=1)

            T_full = kv_latent.shape[1]

            k_nope = self.kv_up_k(kv_latent).view(B, T_full, H, Dk)
            v = self.kv_up_v(kv_latent).view(B, T_full, H, Dk)

            new_cache = (kv_latent, k_rope)
        else:
            new_cache = (kv_latent, k_rope)

        q_nope = q_nope.permute(0, 2, 1, 3).contiguous()
        q_rope = q_rope.permute(0, 2, 1, 3).contiguous()
        k_nope = k_nope.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        k_rope = k_rope.squeeze(2).contiguous()

        out = mla_flash_attention(q_nope, q_rope, k_nope, k_rope, v, self.scale, is_causal=True)
        out = out.permute(0, 2, 1, 3).reshape(B, T, C).to(x.dtype)
        return self.out_proj(out), new_cache


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = int(8 * dim / 3)
        hidden = (hidden + 127) // 128 * 128
        self.w13 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(x).chunk(2, dim=-1)
        return self.drop(self.w2(F.silu(gate) * up))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, lora_rank: int, max_seq_len: int, dropout: float = 0.1, rope_dim: int = 64):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = MultiHeadLatentAttention(dim, heads, max_seq_len=max_seq_len, lora_rank=lora_rank, rope_dim=rope_dim)
        self.ln2 = RMSNorm(dim)
        self.ff = FeedForward(dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        attn_out, new_cache = self.attn(self.ln1(x), offset=offset, kv_cache=kv_cache)
        x_mid = x + attn_out
        out = x_mid + self.ff(self.ln2(x_mid))

        return out, new_cache
