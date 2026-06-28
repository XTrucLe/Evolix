import math
import torch
import triton
import triton.language as tl

LOG2E = math.log2(math.e)


def _mla_prefill_production_configs():
    cfgs = []
    for bt in (32, 64):
        for bs in (32, 64, 128):
            for nw in (4, 8):
                for ns in (2, 3, 4, 5):
                    cfgs.append(triton.Config({"BLOCK_T": bt, "BLOCK_S": bs}, num_warps=nw, num_stages=ns))
    return cfgs


@triton.autotune(configs=_mla_prefill_production_configs(), key=["T", "S", "Dk", "Dr", "Dv", "IS_CAUSAL"])
@triton.jit
def _mla_flash_prefill_production_kernel(
    Q_nope_ptr,
    Q_rope_ptr,
    Kn_ptr,
    Kr_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qrb,
    stride_qrh,
    stride_qrt,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_krb,
    stride_krh,
    stride_krs,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_ob,
    stride_oh,
    stride_ot,
    B,
    H,
    T,
    S,
    Dk: tl.constexpr,
    Dr: tl.constexpr,
    Dv: tl.constexpr,
    scale: tl.constexpr,
    LOG2E: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)

    h = pid_bh % H
    b = pid_bh // H

    t_start = pid_t * BLOCK_T
    t_offs = t_start + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T

    dk_offs = tl.arange(0, Dk)
    dr_offs = tl.arange(0, Dr)
    dv_offs = tl.arange(0, Dv)

    qn_ptr = Q_nope_ptr + b * stride_qb + h * stride_qh + t_offs[:, None] * stride_qt + dk_offs[None, :]
    qr_ptr = Q_rope_ptr + b * stride_qrb + h * stride_qrh + t_offs[:, None] * stride_qrt + dr_offs[None, :]
    qn = tl.load(qn_ptr, mask=t_mask[:, None], other=0.0).to(tl.float16)
    qr = tl.load(qr_ptr, mask=t_mask[:, None], other=0.0).to(tl.float16)

    scale_log2e = scale * LOG2E

    acc = tl.zeros([BLOCK_T, Dv], dtype=tl.float32)
    m_i = tl.full([BLOCK_T], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_T], dtype=tl.float32)

    kn_base = Kn_ptr + b * stride_kb + h * stride_kh
    kr_base = Kr_ptr + b * stride_krb + h * stride_krh
    v_base = V_ptr + b * stride_vb + h * stride_vh

    diagonal_s_idx = t_start // BLOCK_S

    end_s_idx = diagonal_s_idx if IS_CAUSAL else tl.cdiv(S, BLOCK_S)

    for s_idx in range(0, end_s_idx):
        s_start = s_idx * BLOCK_S
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S

        kn = tl.load(kn_base + s_offs[:, None] * stride_ks + dk_offs[None, :], mask=s_mask[:, None], other=0.0).to(tl.float16)
        kr = tl.load(kr_base + s_offs[:, None] * stride_krs + dr_offs[None, :], mask=s_mask[:, None], other=0.0).to(tl.float16)
        v = tl.load(v_base + s_offs[:, None] * stride_vs + dv_offs[None, :], mask=s_mask[:, None], other=0.0).to(tl.float16)

        scores = (tl.dot(qn, tl.trans(kn)) + tl.dot(qr, tl.trans(kr))).to(tl.float32)
        scores = tl.where(t_mask[:, None] & s_mask[None, :], scores, float("-inf"))

        s_scaled = scores * scale_log2e
        m_new = tl.maximum(m_i, tl.max(s_scaled, axis=1))

        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(s_scaled - m_new[:, None])
        p = tl.where(t_mask[:, None] & s_mask[None, :], p, 0.0)

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc + tl.dot(p.to(tl.float16), v).to(tl.float32)
        m_i = m_new

    if IS_CAUSAL and (diagonal_s_idx < tl.cdiv(S, BLOCK_S)):
        s_start = diagonal_s_idx * BLOCK_S
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S

        kn = tl.load(kn_base + s_offs[:, None] * stride_ks + dk_offs[None, :], mask=s_mask[:, None], other=0.0).to(tl.float16)
        kr = tl.load(kr_base + s_offs[:, None] * stride_krs + dr_offs[None, :], mask=s_mask[:, None], other=0.0).to(tl.float16)
        v = tl.load(v_base + s_offs[:, None] * stride_vs + dv_offs[None, :], mask=s_mask[:, None], other=0.0).to(tl.float16)

        scores = (tl.dot(qn, tl.trans(kn)) + tl.dot(qr, tl.trans(kr))).to(tl.float32)

        valid_mask = t_mask[:, None] & s_mask[None, :] & (t_offs[:, None] >= s_offs[None, :])
        scores = tl.where(valid_mask, scores, float("-inf"))

        s_scaled = scores * scale_log2e
        m_new = tl.maximum(m_i, tl.max(s_scaled, axis=1))

        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(s_scaled - m_new[:, None])
        p = tl.where(valid_mask, p, 0.0)

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc + tl.dot(p.to(tl.float16), v).to(tl.float32)
        m_i = m_new

    inv_l = 1.0 / tl.where(l_i > 0.0, l_i, 1.0)
    out_f16 = (acc * inv_l[:, None]).to(tl.float16)
    lse = m_i / LOG2E + tl.log(tl.where(l_i > 0.0, l_i, 1e-8))

    o_ptr = O_ptr + b * stride_ob + h * stride_oh + t_offs[:, None] * stride_ot + dv_offs[None, :]
    tl.store(o_ptr, out_f16, mask=t_mask[:, None])

    lse_ptr = LSE_ptr + (b * H + h) * T + t_offs
    tl.store(lse_ptr, lse, mask=t_mask)


def mla_flash_attention(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool = True,
) -> torch.Tensor:
    B, H, T, Dk = q_nope.shape
    S = k_nope.shape[2]
    Dr = q_rope.shape[3]
    Dv = v.shape[3]

    def prep(t):
        return t.contiguous().to(torch.float16)

    q_nope, q_rope = prep(q_nope), prep(q_rope)
    k_nope, k_rope = prep(k_nope), prep(k_rope)
    v = prep(v)

    out = torch.empty(B, H, T, Dv, device=q_nope.device, dtype=torch.float16)
    lse = torch.empty(B, H, T, device=q_nope.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(T, meta["BLOCK_T"]), B * H)

    _mla_flash_prefill_production_kernel[grid](
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        v,
        out,
        lse,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        q_rope.stride(0),
        q_rope.stride(1),
        q_rope.stride(2),
        k_nope.stride(0),
        k_nope.stride(1),
        k_nope.stride(2),
        k_rope.stride(0),
        k_rope.stride(1),
        k_rope.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B=B,
        H=H,
        T=T,
        S=S,
        Dk=Dk,
        Dr=Dr,
        Dv=Dv,
        scale=scale,
        LOG2E=LOG2E,
        IS_CAUSAL=is_causal,
    )
    return out
