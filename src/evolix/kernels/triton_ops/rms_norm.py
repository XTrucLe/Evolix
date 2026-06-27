import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    X_ptr,
    WN_ptr,
    Out_ptr,
    stride_x_row,
    stride_out_row,
    BT,
    R,
    eps,
    BLOCK_R: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= BT:
        return

    r_offs = tl.arange(0, BLOCK_R)
    r_mask = r_offs < R

    x_ptr = X_ptr + row * stride_x_row + r_offs
    out_ptr = Out_ptr + row * stride_out_row + r_offs

    x = tl.load(x_ptr, mask=r_mask, other=0.0)
    wn = tl.load(WN_ptr + r_offs, mask=r_mask, other=1.0)

    x_squared_masked = tl.where(r_mask, x * x, 0.0)
    sq_mean = tl.sum(x_squared_masked, axis=0) / R

    rms = tl.math.rsqrt(sq_mean + eps)

    out = x * rms * wn
    tl.store(out_ptr, out, mask=r_mask)


def fused_kv_down_rms(
    x: torch.Tensor,
    w_down: torch.Tensor,
    w_norm: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    B, T, C = x.shape
    R = w_down.shape[0]
    BT = B * T

    x_flat = x.reshape(BT, C)
    gemm_out = F.linear(x_flat, w_down).to(torch.float32)

    out = torch.empty(BT, R, device=x.device, dtype=x.dtype)
    BLOCK_R = min(triton.next_power_of_2(R), 1024)

    _rms_norm_kernel[(BT,)](
        gemm_out,
        w_norm.to(torch.float32),
        out,
        gemm_out.stride(0),
        out.stride(0),
        BT,
        R,
        eps,
        BLOCK_R=BLOCK_R,
        num_warps=8,
        num_stages=4,
    )
    return out.view(B, T, R)
