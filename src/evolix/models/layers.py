import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
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
    def __init__(self, dim, heads, max_seq_len=2048, lora_rank=512, rope_dim=64, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        Dv = dim // heads
        assert Dv >= 16 and (Dv & (Dv - 1)) == 0
        assert rope_dim >= 16 and (rope_dim & (rope_dim - 1)) == 0

        self.heads = heads
        self.rope_dim = rope_dim
        self.v_head_dim = Dv
        self.k_head_dim = Dv
        self.qk_dim = Dv + rope_dim
        self.scale = float(self.qk_dim) ** -0.5
        self.lora_rank = lora_rank
        self.dim = dim

        self.q_proj = nn.Linear(dim, heads * self.qk_dim, bias=False)
        self.kv_down = nn.Linear(dim, lora_rank, bias=False)
        self.kv_norm_w = nn.Parameter(torch.ones(lora_rank))
        self.kv_up = nn.Linear(lora_rank, heads * (Dv + Dv), bias=False)
        self.k_rope_proj = nn.Linear(dim, rope_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_rope_cache(max_seq_len)

    def _build_rope_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        emb = torch.cat([torch.outer(t, self.inv_freq)] * 2, dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)
        self._cached_len = seq_len

    def _get_cos_sin(self, T, offset=0):
        needed = offset + T
        if needed > self._cached_len:
            self._build_rope_cache(max(needed, self._cached_len * 2))
        return self.cos_cache[offset : offset + T], self.sin_cache[offset : offset + T]

    def forward(self, x: torch.Tensor, offset: int = 0, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        H, Dk, Dv, Dr = self.heads, self.k_head_dim, self.v_head_dim, self.rope_dim

        q_full = self.q_proj(x).view(B, T, H, Dk + Dr)
        q_nope = q_full[..., :Dk]
        q_rope_ = q_full[..., Dk:]

        kv_latent = fused_kv_down_rms(x, self.kv_down.weight, self.kv_norm_w)

        kv = self.kv_up(kv_latent).view(B, T, H, Dk + Dv)
        k_nope = kv[..., :Dk]
        v = kv[..., Dk:]

        k_rope_shared = self.k_rope_proj(x).view(B, T, 1, Dr)

        cos, sin = self._get_cos_sin(T, offset)
        q_rope_rot, k_rope_rot = fused_rope(q_rope_, k_rope_shared, cos, sin)

        if kv_cache is not None and not self.training:
            lat_prev, kr_prev = kv_cache
            kv_latent_cat = torch.cat([lat_prev, kv_latent], dim=1)
            k_rope_cat = torch.cat([kr_prev, k_rope_rot], dim=1)
            T_full = kv_latent_cat.shape[1]
            kv_full = self.kv_up(kv_latent_cat).view(B, T_full, H, Dk + Dv)
            k_nope = kv_full[..., :Dk]
            v = kv_full[..., Dk:]
            k_rope_use = k_rope_cat
            new_cache = (kv_latent_cat, k_rope_cat)
        else:
            k_rope_use = k_rope_rot
            new_cache = (kv_latent, k_rope_rot)

        q_nope_h = q_nope.permute(0, 2, 1, 3).contiguous()
        q_rope_h = q_rope_rot.permute(0, 2, 1, 3).contiguous()
        k_nope_h = k_nope.permute(0, 2, 1, 3).contiguous()
        v_h = v.permute(0, 2, 1, 3).contiguous()
        k_rope_h = k_rope_use.squeeze(2).contiguous()

        out = mla_flash_attention(
            q_nope_h,
            q_rope_h,
            k_nope_h,
            k_rope_h,
            v_h,
            self.scale,
            is_causal=(T > 1),
        )

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
        self.attn = MultiHeadLatentAttention(dim, heads, max_seq_len=max_seq_len, lora_rank=lora_rank, dropout=dropout, rope_dim=rope_dim)
        self.ln2 = RMSNorm(dim)
        self.ff = FeedForward(dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        use_checkpointing: bool = False,
        offset: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        def _inner(x_in, current_cache):
            attn_out, new_cache = self.attn(self.ln1(x_in), offset=offset, kv_cache=current_cache)
            x_mid = x_in + attn_out
            out = x_mid + self.ff(self.ln2(x_mid))

            return out, new_cache

        if use_checkpointing and self.training:
            return checkpoint(_inner, x, kv_cache, use_reentrant=False)

        return _inner(x, kv_cache)
