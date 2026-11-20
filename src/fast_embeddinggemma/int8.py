"""INT8, the default precision: W8A8 vision products, weights per channel, activations one scale per 64 values (the MLP input:
per row). QKV epilogue: q/k norms + RoPE; gate|up epilogue: gelu * up + INT8 packing. Attention, pooling and the text tower
stay bf16. The kernels assume the vision head size, 64 (half-heads of 32 columns)."""

import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh

from . import kernels as K
from .kernels import round_to

# [Agent] Tile sizes from sweeps on an RTX 3090 at 140 tokens, batch 32; other cards may want others.
TILES = dict(
    rows=dict(BM=4, num_warps=4),
    gate_up=dict(BM=128, BN=64, BK=64, num_warps=4, num_stages=3),
    groups=dict(BM=128, BN=64, num_warps=4, num_stages=3),
    qkv=dict(BM=128, HB=1, BK=64, num_warps=4, num_stages=3),
)


@triton.jit
def quant_groups(u, BM: tl.constexpr, DP: tl.constexpr, G: tl.constexpr):
    # [Agent] One scale per G columns (G = the row width: one per row); 64-value groups stop one large value coarsening the rest.
    s = tl.maximum(tl.max(tl.reshape(tl.abs(u), (BM, DP // G, G)), 2) / 127.0, 1e-8)
    inv = tl.reshape(tl.broadcast_to((1.0 / s)[:, :, None], (BM, DP // G, G)), (BM, DP))
    return tl.minimum(tl.maximum(tl.floor(u * inv + 0.5), -127.0), 127.0).to(tl.int8), s


@triton.jit
def _add_quant(
    o,
    r,
    wp,
    wn,
    ro,
    q,
    f,
    R,
    D: tl.constexpr,
    DP: tl.constexpr,
    BM: tl.constexpr,
    HAS_O: tl.constexpr,
    EPS: tl.constexpr,
    G: tl.constexpr,
    NG: tl.constexpr,
):
    m, k = tl.program_id(0) * BM + tl.arange(0, BM), tl.arange(0, DP)
    ok, at = (m[:, None] < R) & (k[None, :] < D), m[:, None] * D + k[None, :]
    x = tl.load(r + at, ok, 0.0).to(tl.float32)
    if HAS_O:
        b = tl.load(o + at, ok, 0.0).to(tl.float32)
        n = b * tl.rsqrt(tl.sum(b * b, 1) / D + EPS)[:, None] * tl.load(wp + k, k < D, 0.0)[None, :]
        x = (x + n.to(ro.dtype.element_ty).to(tl.float32)).to(ro.dtype.element_ty).to(tl.float32)
    tl.store(ro + at, x.to(ro.dtype.element_ty), ok)
    # [Agent] The next pre-norm is never written: its row statistic rs goes into the factors f = scale * rs, and its
    # weight is multiplied into the row before quantising, so the INT8 product needs no separate norm pass.
    rs = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    qg, s = quant_groups(x * tl.load(wn + k, k < D, 0.0)[None, :], BM, DP, G)
    g = tl.arange(0, DP // G)
    tl.store(q + at, qg, ok)
    tl.store(f + m[:, None] * NG + g[None, :], s * rs[:, None], (m[:, None] < R) & (g[None, :] < NG))


def add_quant(r, o, w_post, w_next, group):
    """r + post-norm(o) (o may be None), and INT8 rows + factors of its next pre-norm: (R, 1) per row, (R, D/64) per group."""
    (R, D), c = r.shape, TILES["rows"]
    DP = triton.next_power_of_2(D)
    G, ng = (64, D // 64) if group else (DP, 1)
    ro, q, f = (
        torch.empty_like(r),
        torch.empty(R, D, dtype=torch.int8, device=r.device),
        torch.empty(R, ng, device=r.device),
    )
    # [Agent] With no branch (o None) the kernel never reads o or wp: r and w_next only fill their pointer slots.
    o_slot, wp_slot = r if o is None else o, w_next if w_post is None else w_post
    _add_quant[(triton.cdiv(R, c["BM"]),)](
        o_slot, r, wp_slot, w_next, ro, q, f, R, D=D, DP=DP, HAS_O=o is not None, EPS=K.EPS, G=G, NG=ng, **c
    )
    return ro, q, f


@triton.jit
def _quant_rows(x, q, f, R, D: tl.constexpr, DP: tl.constexpr, BM: tl.constexpr, G: tl.constexpr, NG: tl.constexpr):
    m, k = tl.program_id(0) * BM + tl.arange(0, BM), tl.arange(0, DP)
    ok, at = (m[:, None] < R) & (k[None, :] < D), m[:, None] * D + k[None, :]
    qg, s = quant_groups(tl.load(x + at, ok, 0.0).to(tl.float32), BM, DP, G)
    g = tl.arange(0, DP // G)
    tl.store(q + at, qg, ok)
    tl.store(f + m[:, None] * NG + g[None, :], s, (m[:, None] < R) & (g[None, :] < NG))


def quant_rows(x):
    """bf16 rows (the attention output) -> INT8 rows + one scale per 64 columns."""
    (R, D), c = x.shape, TILES["rows"]
    q, f = torch.empty(R, D, dtype=torch.int8, device=x.device), torch.empty(R, D // 64, device=x.device)
    _quant_rows[(triton.cdiv(R, c["BM"]),)](x, q, f, R, D=D, DP=triton.next_power_of_2(D), G=64, NG=D // 64, **c)
    return q, f


@triton.jit
def _gate_up(a, f, w, sw, wu, su, c, gs, R, N, D, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m, n, k = tl.program_id(1) * BM + tl.arange(0, BM), tl.program_id(0) * BN + tl.arange(0, BN), tl.arange(0, BK)
    acc, acu = tl.zeros([BM, BN], tl.int32), tl.zeros([BM, BN], tl.int32)
    for k0 in range(0, D, BK):
        x = tl.load(a + m[:, None] * D + (k0 + k)[None, :], (m < R)[:, None], 0)
        # [Agent] The weight tile is loaded [BN, BK] and transposed in registers: a [BK, BN] load reads it byte by byte.
        acc = tl.dot(x, tl.trans(tl.load(w + n[:, None] * D + (k0 + k)[None, :])), acc)
        acu = tl.dot(x, tl.trans(tl.load(wu + n[:, None] * D + (k0 + k)[None, :])), acu)
    fr = tl.load(f + m, m < R, 0.0)[:, None]
    y = acc.to(tl.float32) * fr * tl.load(sw + n)[None, :]
    act = 0.5 * y * (1.0 + tanh(0.7978845608028654 * (y + 0.044715 * (y * y * y))))
    y = act * (acu.to(tl.float32) * fr * tl.load(su + n)[None, :])
    # [Agent] The tile holds 64 columns of each row: one INT8 scale per (row, tile), so the MLP middle is written once as INT8.
    yq, sc = quant_groups(y, BM, BN, BN)
    tl.store(gs + m * (N // BN) + tl.program_id(0), tl.reshape(sc, (BM,)), m < R)
    tl.store(c + m[:, None] * N + n[None, :], yq, (m < R)[:, None])


def gate_up(q, f, w, sw, wu, su):
    """INT8 rows + row factor x INT8 gate and up -> gelu(gate) * up as INT8 (R, N) + one scale per (row, 64 columns)."""
    (R, D), N, c = q.shape, w.shape[0], TILES["gate_up"]
    assert D % c["BK"] == 0 and N % c["BN"] == 0
    y, gs = torch.empty(R, N, dtype=torch.int8, device=q.device), torch.empty(R, N // c["BN"], device=q.device)
    _gate_up[(N // c["BN"], triton.cdiv(R, c["BM"]))](q, f, w, sw, wu, su, y, gs, R, N, D, **c)
    return y, gs


@triton.jit
def _gemm_groups(a, s, w, sw, c, R, N, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, G: tl.constexpr):
    m, n, k = tl.program_id(1) * BM + tl.arange(0, BM), tl.program_id(0) * BN + tl.arange(0, BN), tl.arange(0, G)
    acc = tl.zeros([BM, BN], tl.float32)
    # [Agent] One K step = one 64-column group of the input: its int32 partial product is scaled by that group's row scale
    # before it joins the fp32 sum; the weight's channel scale waits for the epilogue.
    for g in range(0, D // G):
        x = tl.load(a + m[:, None] * D + (g * G + k)[None, :], (m < R)[:, None], 0)
        p = tl.dot(x, tl.trans(tl.load(w + n[:, None] * D + (g * G + k)[None, :])), out_dtype=tl.int32)
        acc += p.to(tl.float32) * tl.load(s + m * (D // G) + g, m < R, 0.0)[:, None]
    tl.store(c + m[:, None] * N + n[None, :], (acc * tl.load(sw + n)[None, :]).to(c.dtype.element_ty), (m < R)[:, None])


def gemm_groups(q, s, w, sw):
    """INT8 input with one scale per (row, 64 columns) x INT8 weight (N, D) with channel scales -> bf16 (R, N)."""
    (R, D), N, c = q.shape, w.shape[0], TILES["groups"]
    y = torch.empty(R, N, dtype=torch.bfloat16, device=q.device)
    _gemm_groups[(N // c["BN"], triton.cdiv(R, c["BM"]))](q, s, w, sw, y, R, N, D=D, G=D // s.shape[1], **c)
    return y


@triton.jit
def head_sum(x, BM: tl.constexpr, HB: tl.constexpr):
    s = tl.sum(tl.reshape(x, (BM, HB, 32)), 2)  # each entry -> the sum over its 32-column half-head
    return tl.reshape(tl.broadcast_to(s[:, :, None], (BM, HB, 32)), (BM, HB * 32))


@triton.jit
def _qkv_rope(
    a,
    f,
    wa,
    wb,
    sa,
    sb,
    na,
    nb,
    cs,
    out,
    R,
    L,
    D: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr,
    HB: tl.constexpr,
    BK: tl.constexpr,
    EPS: tl.constexpr,
):
    # [Agent] The weight rows are split per head into A = channels with bit 16 clear and B = their RoPE partners (c ^ 16), so a
    # channel and its partner sit at the same place of two accumulators and norm + RoPE run element-wise in the epilogue.
    pn, m, k = tl.program_id(0), tl.program_id(1) * BM + tl.arange(0, BM), tl.arange(0, BK)
    n, ok = pn * HB * 32 + tl.arange(0, HB * 32), (tl.program_id(1) * BM + tl.arange(0, BM) < R)[:, None]
    acc_a, acc_b = tl.zeros([BM, HB * 32], tl.float32), tl.zeros([BM, HB * 32], tl.float32)
    for k0 in range(0, D, BK):
        x = tl.load(a + m[:, None] * D + (k0 + k)[None, :], ok, 0)
        # [Agent] BK = 64 = one group: each K step is scaled by its group factor before the fp32 sum (exact: |sum| < 2^24).
        fg = tl.load(f + m * (D // BK) + k0 // BK, m < R, 0.0)[:, None]
        wt = n[:, None] * D + (k0 + k)[None, :]
        acc_a += tl.dot(x, tl.trans(tl.load(wa + wt)), out_dtype=tl.int32).to(tl.float32) * fg
        acc_b += tl.dot(x, tl.trans(tl.load(wb + wt)), out_dtype=tl.int32).to(tl.float32) * fg
    xa = round_to(acc_a * tl.load(sa + n)[None, :], True)
    xb = round_to(acc_b * tl.load(sb + n)[None, :], True)
    inv = tl.rsqrt((head_sum(xa * xa, BM, HB) + head_sum(xb * xb, BM, HB)) / 64 + EPS)
    ya, yb = round_to(xa * inv * tl.load(na + n)[None, :], True), round_to(xb * inv * tl.load(nb + n)[None, :], True)
    j = n % 32
    ca = (n // 32) * 64 + j + (j & 16)  # A channel of column n; its partner is ca + 16
    if pn * HB < 2 * H:  # q and k heads turn, v heads do not
        t = (m % L)[:, None] * 64 + (j + (j & 16))[None, :]
        cos_a, sin_a = tl.load(cs + t).to(tl.float32), tl.load(cs + L * 64 + t).to(tl.float32)
        cos_b, sin_b = tl.load(cs + t + 16).to(tl.float32), tl.load(cs + L * 64 + t + 16).to(tl.float32)
        # [Agent] Eager rounds each product and the sum to bf16 (as prep_qkv does): kept.
        ya, yb = (
            round_to(round_to(ya * cos_a, True) - round_to(yb * sin_a, True), True),
            round_to(round_to(yb * cos_b, True) + round_to(ya * sin_b, True), True),
        )
    tl.store(out + m[:, None] * (3 * H * 64) + ca[None, :], ya.to(tl.bfloat16), ok)
    tl.store(out + m[:, None] * (3 * H * 64) + ca[None, :] + 16, yb.to(tl.bfloat16), ok)


def qkv_rope(q, f, s, cs):
    """INT8 rows (R, D) + group factors (R, D/64) -> bf16 q|k|v (R, 3 H 64), q/k normed and turned; s = split_heads(weight)."""
    (R, D), H, c = q.shape, q.shape[1] // 64, TILES["qkv"]
    out = torch.empty(R, 3 * H * 64, dtype=torch.bfloat16, device=q.device)
    _qkv_rope[(3 * H // c["HB"], triton.cdiv(R, c["BM"]))](q, f, *s, cs, out, R, cs.shape[1], D=D, H=H, EPS=K.EPS, **c)
    return out


def split_heads(w8, sw, qk):
    """INT8 q|k|v weight + channel scales + norm rows (q, k, ones) -> the A rows, B (partner) rows, scales and norms."""
    j = torch.arange(32, device=w8.device)
    rows = (torch.arange(w8.shape[0] // 64, device=w8.device)[:, None] * 64 + j + (j & 16)).flatten()
    kind, c = rows // (w8.shape[0] // 3), rows % 64
    return tuple(
        t.contiguous() for t in (w8[rows], w8[rows + 16], sw[rows], sw[rows + 16], qk[kind, c], qk[kind, c + 16])
    )


def quantize_weight(w):  # (N, K) -> INT8 (N, K), fp32 channel scale (N,)
    s = (w.float().abs().amax(1) / 127.0).clamp_min(1e-8)
    return (w.float() / s[:, None] + 0.5).floor().clamp(-127, 127).to(torch.int8).contiguous(), s.contiguous()


def quantize(W):
    """Vision weights with INT8 products (replacing the bf16 ones); patch, position and embedder weights stay bf16."""

    def layer(lw):
        g, u = lw["gu"].chunk(2)
        return lw | dict(
            qkv=split_heads(*quantize_weight(lw["qkv"]), lw["qk"]),
            wo=quantize_weight(lw["wo"]),
            wd=quantize_weight(lw["wd"]),
            gu=quantize_weight(g) + quantize_weight(u),
        )

    return W | dict(layers=[layer(lw) for lw in W["layers"]])


def run_layer(h, x, lw, w_next, cs):
    """model.run_layer with INT8 products; x = (INT8 rows, 64-value group factors) of the pre-norm input of the QKV product."""
    H = h.shape[1] // 64
    a = K.attend(qkv_rope(*x, lw["qkv"], cs), cs.shape[1], H, H, 64)
    # [Agent] The MLP input gets one scale per row (group False): its product is the fused gate|up kernel.
    h, q, f = add_quant(h, gemm_groups(*quant_rows(a), *lw["wo"]), lw["w_pa"], lw["w_pre"], False)
    # [Agent] The last layer has no next pre-norm; its INT8 rows are never read, so any norm scale fills the slot.
    h, q, f = add_quant(
        h, gemm_groups(*gate_up(q, f, *lw["gu"]), *lw["wd"]), lw["w_pf"], lw["w_pf"] if w_next is None else w_next, True
    )
    return h, (q, f)
