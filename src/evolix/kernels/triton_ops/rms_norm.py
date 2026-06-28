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
    R,
    eps,
    BLOCK_R: tl.constexpr,
):
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_R)
    mask = offs < R

    x_ptr = X_ptr + row * stride_x_row + offs
    w_ptr = WN_ptr + offs
    out_ptr = Out_ptr + row * stride_out_row + offs

    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

    x2 = x * x
    mean = tl.sum(x2 * mask, axis=0) / R
    inv_rms = tl.math.rsqrt(mean + eps)

    w = tl.load(w_ptr, mask=mask, other=1.0).to(tl.float32)

    out = x * inv_rms * w

    tl.store(out_ptr, out, mask=mask)


def fused_kv_down_rms(x, w_down, w_norm, eps=1e-6):
    B, T, C = x.shape
    R = w_down.shape[0]
    BT = B * T

    x_flat = x.reshape(BT, C)

    gemm_out = F.linear(x_flat, w_down).contiguous().to(torch.float32)
    out = torch.empty((BT, R), device=x.device, dtype=torch.float32)
    BLOCK_R = triton.next_power_of_2(min(R, 1024))

    _rms_norm_kernel[(BT,)](
        gemm_out,
        w_norm,
        out,
        gemm_out.stride(0),
        out.stride(0),
        R,
        eps,
        BLOCK_R=BLOCK_R,
        num_warps=4,
        num_stages=2,
    )

    return out.view(B, T, R).to(x.dtype)
