import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from torch.nn.attention import sdpa_kernel, SDPBackend

BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


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

        self.q_nope = nn.Linear(dim, heads * self.head_dim, bias=False)

        self.kv_down = nn.Linear(dim, lora_rank, bias=False)
        self.kv_up_k = nn.Linear(lora_rank, heads * self.head_dim, bias=False)
        self.kv_up_v = nn.Linear(lora_rank, heads * self.head_dim, bias=False)
        self.kv_norm_w = nn.Parameter(torch.ones(lora_rank))

        self.out_proj = nn.Linear(dim, dim, bias=False)

        inv_freq = 1.0 / (12800.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
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

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, offset=0, kv_cache=None):
        B, T, C = x.shape
        H = self.heads
        Dk = self.head_dim

        q = self.q_nope(x).view(B, T, H, Dk)
        cos, sin = self._get_cos_sin(T, offset)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q_rot = q * cos + self._rotate_half(q) * sin

        kv_latent = F.linear(x, self.kv_down.weight)
        kv_latent = F.rms_norm(kv_latent, (kv_latent.shape[-1],), self.kv_norm_w, 1e-6)

        if kv_cache is not None and not self.training:
            lat_prev, _ = kv_cache
            kv_latent = torch.cat([lat_prev, kv_latent], dim=1)
            T_full = kv_latent.shape[1]
        else:
            T_full = T

        k = self.kv_up_k(kv_latent).view(B, T_full, H, Dk)
        v = self.kv_up_v(kv_latent).view(B, T_full, H, Dk)

        cos_full = self.cos_cache[offset : offset + T_full].unsqueeze(0).unsqueeze(2)
        sin_full = self.sin_cache[offset : offset + T_full].unsqueeze(0).unsqueeze(2)
        k_rot = k * cos_full + self._rotate_half(k) * sin_full

        q_attn = q_rot.permute(0, 2, 1, 3).contiguous()
        k_attn = k_rot.permute(0, 2, 1, 3).contiguous()
        v_attn = v.permute(0, 2, 1, 3).contiguous()

        with sdpa_kernel(backends=BACKENDS):
            out = F.scaled_dot_product_attention(q_attn, k_attn, v_attn, is_causal=True, dropout_p=0.0, scale=self.scale)

        out = out.permute(0, 2, 1, 3).reshape(B, T, C).to(x.dtype)
        out = self.out_proj(out)

        new_cache = (kv_latent, None) if not self.training else None
        return out, new_cache


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = int(8 * dim / 3)
        hidden = (hidden + 63) // 64 * 64
        self.w13 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(x).chunk(2, dim=-1)
        return self.drop(self.w2(F.silu(gate) * up))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, lora_rank: int, max_seq_len: int, dropout: float = 0.1, rope_dim: int = 64):
        super().__init__()
        self.ln1 = nn.RMSNorm(dim)
        self.attn = MultiHeadLatentAttention(dim, heads, max_seq_len=max_seq_len, lora_rank=lora_rank, rope_dim=rope_dim)
        self.ln2 = nn.RMSNorm(dim)
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
