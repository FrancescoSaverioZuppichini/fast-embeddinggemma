"""Attention of both towers: the fused per-head q/k norm + RoPE kernel on the q|k|v product, then bidirectional SDPA."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .norm import EPS, round_to


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


def attend(qkv, L, hq, hkv, hd, mask=None):
    """Prepared q|k|v rows (N, (hq + 2 hkv) hd) -> attention output (N, hq hd): bidirectional, scale 1.0; mask (N // L, L) bool,
    True = a real key (padded search queries), None = no padding (pictures)."""
    N = qkv.shape[0]
    q, k, v = (t.view(N // L, L, -1, hd).transpose(1, 2) for t in qkv.split((hq * hd, hkv * hd, hkv * hd), -1))
    # [Agent] Unmasked, SDPA shares the key heads itself (bit-identical to repeating them, 0.34 ms faster per batch at head 512).
    # With a key mask k/v are repeated, as HF does: that is the path the query vectors were checked against.
    gqa = hq != hkv and mask is None
    if hq != hkv and not gqa:
        k, v = k.repeat_interleave(hq // hkv, 1), v.repeat_interleave(hq // hkv, 1)
    keys = None if mask is None else mask[:, None, None, :]
    return (
        F.scaled_dot_product_attention(q, k, v, attn_mask=keys, scale=1.0, enable_gqa=gqa)
        .transpose(1, 2)
        .reshape(N, hq * hd)
    )
