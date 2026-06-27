import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def _rope_q(Q_ptr, Cos_ptr, Sin_ptr, Q_out_ptr, T, H, Dr: tl.constexpr, HALF: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // (T * H)
    th = pid % (T * H)
    t = th // H
    h = th % H
    if t >= T:
        return

    ho = tl.arange(0, HALF)

    cos1 = tl.load(Cos_ptr + t * Dr + ho)
    cos2 = tl.load(Cos_ptr + t * Dr + HALF + ho)
    sin1 = tl.load(Sin_ptr + t * Dr + ho)
    sin2 = tl.load(Sin_ptr + t * Dr + HALF + ho)

    base = Q_ptr + (b * T * H + t * H + h) * Dr
    x1 = tl.load(base + ho).to(tl.float32)
    x2 = tl.load(base + HALF + ho).to(tl.float32)

    tl.store(Q_out_ptr + (b * T * H + t * H + h) * Dr + ho, (x1 * cos1 - x2 * sin1).to(tl.float16))
    tl.store(Q_out_ptr + (b * T * H + t * H + h) * Dr + HALF + ho, (x2 * cos2 + x1 * sin2).to(tl.float16))


@triton.jit
def _rope_k(K_ptr, Cos_ptr, Sin_ptr, K_out_ptr, T, Dr: tl.constexpr, HALF: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // T
    t = pid % T
    if t >= T:
        return

    ho = tl.arange(0, HALF)
    cos1 = tl.load(Cos_ptr + t * Dr + ho)
    cos2 = tl.load(Cos_ptr + t * Dr + HALF + ho)
    sin1 = tl.load(Sin_ptr + t * Dr + ho)
    sin2 = tl.load(Sin_ptr + t * Dr + HALF + ho)

    base = K_ptr + (b * T + t) * Dr
    x1 = tl.load(base + ho).to(tl.float32)
    x2 = tl.load(base + HALF + ho).to(tl.float32)

    tl.store(K_out_ptr + (b * T + t) * Dr + ho, (x1 * cos1 - x2 * sin1).to(tl.float16))
    tl.store(K_out_ptr + (b * T + t) * Dr + HALF + ho, (x2 * cos2 + x1 * sin2).to(tl.float16))


def fused_rope(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, Dr = q_rope.shape
    HALF = Dr // 2
    q_out = torch.empty_like(q_rope)
    k_sq = k_rope.squeeze(2).contiguous()
    k_out = torch.empty_like(k_sq)

    _rope_q[(B * T * H,)](
        q_rope.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        q_out,
        T,
        H,
        Dr=Dr,
        HALF=HALF,
    )
    _rope_k[(B * T,)](
        k_sq,
        cos.contiguous(),
        sin.contiguous(),
        k_out,
        T,
        Dr=Dr,
        HALF=HALF,
    )
    return q_out, k_out.unsqueeze(2)
