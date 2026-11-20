"""Ops of both towers: two plain PyTorch ops and three Triton kernels. A kernel rounds where eager stores a tensor, to
the output dtype (bf16: eager's rounding points; fp32: none), so one code runs the bf16 engine and the fp32 exactness test."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh

EPS = 1e-6  # rms_norm_eps of both towers


def rms_norm(x, w=None):
    """HF RMSNorm: statistics in fp32, optional fp32 scale, result in the input dtype."""
    xf = x.float()
    y = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + EPS, -0.5)
    return (y if w is None else y * w).to(x.dtype)


def attend(qkv, L, hq, hkv, hd):
    """Prepared q|k|v rows (N, (hq + 2 hkv) hd) -> attention output (N, hq hd): bidirectional, scale 1.0, no mask."""
    N = qkv.shape[0]
    q, k, v = (t.view(N // L, L, -1, hd).transpose(1, 2) for t in qkv.split((hq * hd, hkv * hd, hkv * hd), -1))
    # [Agent] GQA exactly as HF decides it: SDPA's own GQA up to head 256 (flash), repeated k/v beyond (head 512 has no flash).
    gqa = hq != hkv and hd <= 256
    if hq != hkv and not gqa:
        k, v = k.repeat_interleave(hq // hkv, 1), v.repeat_interleave(hq // hkv, 1)
    return F.scaled_dot_product_attention(q, k, v, scale=1.0, enable_gqa=gqa).transpose(1, 2).reshape(N, hq * hd)


@triton.jit
def round_to(x, BF: tl.constexpr):
    # [Agent] Round fp32 to bf16 with ties to even but keep it fp32, so each eager rounding point is explicit; a bf16 cast
    # would truncate in the CPU interpreter. With fp32 storage (BF off) nothing is rounded.
    if BF:
        b = x.to(tl.int32, bitcast=True)
        x = ((b + 0x7FFF + ((b >> 16) & 1)) & -65536).to(tl.float32, bitcast=True)
    return x


@triton.jit
def _prep_qkv(
    qkv,
    cs,
    w,
    out,
    N,
    L,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    HEAD: tl.constexpr,
    HALF: tl.constexpr,
    BT: tl.constexpr,
    BF: tl.constexpr,
    EPS: tl.constexpr,
):
    tile, head = tl.program_id(0), tl.program_id(1)
    kind = (head >= HQ).to(tl.int32) + (head >= HQ + HKV).to(tl.int32)  # 0 q, 1 k, 2 v
    rows, c = tile * BT + tl.arange(0, BT), tl.arange(0, HEAD)
    ok, at = (rows < N)[:, None], rows[:, None] * ((HQ + 2 * HKV) * HEAD) + head * HEAD
    # [Agent] RoPE turns channel c with its partner c ^ HALF (HF rotate_half: 16 inside each 32-channel half for the 2-D
    # vision RoPE, head/2 for text), so the partner is loaded and normed beside it. cs is (2, L, HEAD): cos then sin.
    x = tl.load(qkv + at + c[None, :], ok, 0.0).to(tl.float32)
    xp = tl.load(qkv + at + (c ^ HALF)[None, :], ok, 0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / HEAD + EPS)[:, None]
    y = round_to(x * inv * tl.load(w + kind * HEAD + c)[None, :], BF)
    yp = round_to(xp * inv * tl.load(w + kind * HEAD + (c ^ HALF))[None, :], BF)
    if kind < 2:
        t = (rows % L)[:, None] * HEAD + c[None, :]
        cos, sin = tl.load(cs + t).to(tl.float32), tl.load(cs + L * HEAD + t).to(tl.float32)
        # [Agent] Eager rounds each product and the sum to bf16; so does this (bit-exact in all 16 vision layers).
        y = round_to(round_to(y * cos, BF) + round_to(tl.where((c & HALF) == 0, -1.0, 1.0)[None, :] * yp * sin, BF), BF)
    tl.store(out + at + c[None, :], y.to(out.dtype.element_ty), ok)


def prep_qkv(qkv, cs, w, hq, hkv, half):
    """q|k|v projections (N, (hq + 2 hkv) head) -> per-head RMSNorm (w rows: q, k, ones for v), RoPE on q and k."""
    N, L, hd = qkv.shape[0], cs.shape[1], cs.shape[2]
    out, bt = torch.empty_like(qkv), 1024 // hd
    _prep_qkv[(triton.cdiv(N, bt), hq + 2 * hkv)](
        qkv, cs, w, out, N, L, HQ=hq, HKV=hkv, HEAD=hd, HALF=half, BT=bt, BF=qkv.dtype == torch.bfloat16, EPS=EPS
    )
    return out


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


def add_norm(h, branch, w_post, w_next, scale=None):
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


@triton.jit
def _gelu_up(gp, up, out, GS, US, FF: tl.constexpr, BLOCK: tl.constexpr, BF: tl.constexpr):
    row, c = tl.program_id(0), tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(gp + row * GS + c, c < FF, 0.0).to(tl.float32)
    u = tl.load(up + row * US + c, c < FF, 0.0).to(tl.float32)
    # [Agent] torch's own gelu_tanh term by term (same constants, same order, same libdevice tanh), so the bf16 result is
    # eager's; an exp form was as accurate but differed from eager in 0.1-7% of entries.
    act = round_to(0.5 * g * (1.0 + tanh(0.7978845608028654 * (g + 0.044715 * (g * g * g)))), BF)
    tl.store(out + row * FF + c, round_to(act * u, BF).to(out.dtype.element_ty), c < FF)


def gelu_up(g, u):
    """gelu_tanh(g) * u for two (N, FF) row-strided views (the halves of a merged gate|up product, or a PLE gate and slice)."""
    N, FF = g.shape
    out = torch.empty(N, FF, dtype=g.dtype, device=g.device)
    _gelu_up[(N, triton.cdiv(FF, 1024))](
        g, u, out, g.stride(0), u.stride(0), FF=FF, BLOCK=1024, BF=g.dtype == torch.bfloat16
    )
    return out
