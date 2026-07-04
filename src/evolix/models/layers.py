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
        self.nope_dim = self.head_dim - rope_dim
        self.lora_rank = lora_rank
        self.max_seq_len = max_seq_len
        self.scale = self.head_dim**-0.5

        self.q_nope = nn.Linear(dim, heads * self.nope_dim, bias=False)
        self.q_rope = nn.Linear(dim, heads * rope_dim, bias=False)
        self.kv_down = nn.Linear(dim, lora_rank, bias=False)
        self.kv_norm_w = nn.Parameter(torch.ones(lora_rank))
        self.kv_up_k_nope = nn.Linear(lora_rank, heads * self.nope_dim, bias=False)
        self.k_rope = nn.Linear(dim, rope_dim, bias=False)
        self.kv_up_v = nn.Linear(lora_rank, heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cache", emb.sin()[None, :, None, :], persistent=False)

        self.register_buffer("_wuk", None, persistent=False)
        self.register_buffer("_wuv", None, persistent=False)

    def train(self, mode: bool = True):
        if mode:
            self._wuk = None
            self._wuv = None
        return super().train(mode)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, x, cos, sin):
        return x * cos + self._rotate_half(x) * sin

    def build_absorbed_weights(self):
        H, Dn, Dh, R = self.heads, self.nope_dim, self.head_dim, self.lora_rank
        self._wuk = self.kv_up_k_nope.weight.detach().view(H, Dn, R).contiguous()
        self._wuv = self.kv_up_v.weight.detach().view(H, Dh, R).contiguous()

    def init_cache(self, batch_size: int, device, dtype):
        latent_cache = torch.zeros(batch_size, self.max_seq_len, self.lora_rank, device=device, dtype=dtype)
        rope_cache = torch.zeros(batch_size, self.max_seq_len, self.rope_dim, device=device, dtype=dtype)
        return latent_cache, rope_cache

    def forward(self, x, offset: int = 0, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if self.training or kv_cache is None:
            return self._forward_train(x, offset)
        return self._forward_decode(x, offset, kv_cache)

    def _forward_train(self, x, offset):
        B, T, C = x.shape
        H, Dr, Dn = self.heads, self.rope_dim, self.nope_dim

        q_nope = self.q_nope(x).view(B, T, H, Dn)
        q_rope = self.q_rope(x).view(B, T, H, Dr)

        cos = self.cos_cache[:, offset : offset + T]
        sin = self.sin_cache[:, offset : offset + T]
        q_rope = self._apply_rope(q_rope, cos, sin)
        q_full = torch.cat([q_nope, q_rope], dim=-1)

        latent = self.kv_down(x)
        latent = F.rms_norm(latent, (self.lora_rank,), self.kv_norm_w, 1e-6)

        k_rope = self.k_rope(x).view(B, T, 1, Dr)
        k_rope = self._apply_rope(k_rope, cos, sin).expand(-1, -1, H, -1)

        k_nope = self.kv_up_k_nope(latent).view(B, T, H, Dn)
        v = self.kv_up_v(latent).view(B, T, H, self.head_dim)
        k_full = torch.cat([k_nope, k_rope], dim=-1)

        q_attn = q_full.permute(0, 2, 1, 3)
        k_attn = k_full.permute(0, 2, 1, 3)
        v_attn = v.permute(0, 2, 1, 3)

        with sdpa_kernel(backends=BACKENDS):
            out = F.scaled_dot_product_attention(q_attn, k_attn, v_attn, is_causal=True, dropout_p=0.0, scale=self.scale)

        out = out.transpose(1, 2).reshape(B, T, C).to(x.dtype)
        return self.out_proj(out), None

    def _forward_decode(self, x, offset: int, kv_cache: Tuple[torch.Tensor, torch.Tensor]):
        if self._wuk is None:
            self.build_absorbed_weights()

        B, T, C = x.shape
        H, Dr, Dn, R, S = self.heads, self.rope_dim, self.nope_dim, self.lora_rank, self.max_seq_len
        latent_cache, rope_cache = kv_cache

        q_nope = self.q_nope(x).view(B, T, H, Dn)
        q_rope = self.q_rope(x).view(B, T, H, Dr)

        cos = self.cos_cache[:, offset : offset + T]
        sin = self.sin_cache[:, offset : offset + T]
        q_rope = self._apply_rope(q_rope, cos, sin)

        q_absorbed = torch.einsum("bthd,hdr->bthr", q_nope, self._wuk)

        latent_new = self.kv_down(x)
        latent_new = F.rms_norm(latent_new, (R,), self.kv_norm_w, 1e-6)
        k_rope_new = self._apply_rope(self.k_rope(x).view(B, T, 1, Dr), cos, sin).squeeze(2)

        pos = torch.arange(offset, offset + T, device=x.device)
        latent_cache.index_copy_(1, pos, latent_new)
        rope_cache.index_copy_(1, pos, k_rope_new)

        scores_nope = torch.einsum("bthr,bsr->bhts", q_absorbed, latent_cache)
        scores_rope = torch.einsum("bthd,bsd->bhts", q_rope, rope_cache)
        scores = (scores_nope + scores_rope) * self.scale

        key_pos = torch.arange(S, device=x.device)
        causal_mask = key_pos[None, :] > pos[:, None]
        scores = scores.masked_fill(causal_mask[None, None], float("-inf"))

        attn = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        al = torch.einsum("bhts,bsr->bhtr", attn, latent_cache)
        out = torch.einsum("bhtr,hdr->bthd", al, self._wuv).reshape(B, T, C)

        return self.out_proj(out), (latent_cache, rope_cache)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = int(8 * dim / 3 + 127) // 128 * 128
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
        x = x + attn_out
        ff_out = self.ff(self.ln2(x))
        out = x + ff_out
        return out, new_cache
