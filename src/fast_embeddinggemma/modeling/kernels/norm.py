"""RMS norms of both towers: the plain op, the fused residual add + post-norm (+ the next pre-norm) kernel, and round_to,
the bf16 rounding every kernel uses so one code runs the bf16 engine and the fp32 exactness test."""

import torch
import triton
import triton.language as tl

EPS = 1e-6  # rms_norm_eps of both towers


def rms_norm(x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """HF RMSNorm: statistics in fp32, optional fp32 scale, result in the input dtype."""
    xf = x.float()
    y = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + EPS, -0.5)
    return (y if w is None else y * w).to(x.dtype)


@triton.jit
def round_to(x, BF: tl.constexpr):
    # [Agent] Round fp32 to bf16 with ties to even but keep it fp32, so each eager rounding point is explicit; a bf16 cast
    # would truncate in the CPU interpreter. With fp32 storage (BF off) nothing is rounded.
    if BF:
        b = x.to(tl.int32, bitcast=True)
        x = ((b + 0x7FFF + ((b >> 16) & 1)) & -65536).to(tl.float32, bitcast=True)
    return x


@triton.jit
def _add_norm(
    h,
    br,
    wp,
    wn,
    sc,
    ho,
    x,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    NEXT: tl.constexpr,
    BF: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
):
    row, c = tl.program_id(0), tl.arange(0, BLOCK)
    ok, at = c < D, row * D + c
    b = tl.load(br + at, ok, 0.0).to(tl.float32)
    n = round_to(b * tl.rsqrt(tl.sum(b * b, 0) / D + EPS) * tl.load(wp + c, ok, 0.0), BF)
    hn = round_to(tl.load(h + at, ok, 0.0).to(tl.float32) + n, BF)
    if SCALE:  # text layer end: times the layer scalar
        hn = round_to(hn * tl.load(sc).to(tl.float32), BF)
    tl.store(ho + at, hn.to(ho.dtype.element_ty), ok)
    if NEXT:
        inv = tl.rsqrt(tl.sum(hn * hn, 0) / D + EPS)
        tl.store(x + at, round_to(hn * inv * tl.load(wn + c, ok, 0.0), BF).to(x.dtype.element_ty), ok)


def add_norm(
    h: torch.Tensor,
    branch: torch.Tensor,
    w_post: torch.Tensor,
    w_next: torch.Tensor | None,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """h + post-norm(branch) (times scale), and the next sublayer's pre-norm of it (None when no norm follows)."""
    ho, D, nxt = torch.empty_like(h), h.shape[1], w_next is not None
    # [Agent] With NEXT / SCALE off, x, wn and sc are never read or written; ho and w_post only fill their pointer slots.
    x = torch.empty_like(h) if nxt else ho
    _add_norm[(h.shape[0],)](
        h,
        branch,
        w_post,
        w_next if nxt else w_post,
        w_post if scale is None else scale,
        ho,
        x,
        D=D,
        BLOCK=triton.next_power_of_2(D),
        NEXT=nxt,
        BF=h.dtype == torch.bfloat16,
        EPS=EPS,
        SCALE=scale is not None,
    )
    return ho, x if nxt else None
