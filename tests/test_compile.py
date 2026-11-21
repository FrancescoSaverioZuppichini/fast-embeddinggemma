"""Every Triton kernel, in every configuration the engine launches, compiled for sm_86 (RTX 30xx) with no card: Triton's CPU
interpreter accepts code the GPU compiler rejects, so this runs before any card time."""

import pytest
import triton
from triton.backends.compiler import GPUTarget

from fast_embeddinggemma.modeling import attention, mlp, norm
from fast_embeddinggemma.modeling import int8 as I

BF, F32, I8, EPS = "*bf16", "*fp32", "*i8", norm.EPS


def launch(tile):
    return {k: tile[k] for k in ("num_warps", "num_stages") if k in tile}


def configs():
    """(kernel, runtime argument types, constexpr arguments, launch options) of every launch in the engine."""
    for dt, bf in ((BF, True), (F32, False)):
        p = dict(qkv=dt, cs=dt, w=F32, out=dt, N="i32", L="i32")
        for hq, hkv, hd in ((12, 12, 64), (4, 2, 256), (4, 1, 512)):
            half = hd // 4 if hq == hkv else hd // 2
            yield attention._qkv_rope, p, dict(HQ=hq, HKV=hkv, HEAD=hd, HALF=half, BT=1024 // hd, BF=bf, EPS=EPS), {}
        for d, nxt, scale in (
            (768, True, False),
            (768, False, False),
            (512, True, False),
            (512, False, False),
            (512, True, True),
        ):
            p = dict(h=dt, br=dt, wp=F32, wn=F32, sc=dt if scale else F32, ho=dt, x=dt)
            yield (
                norm._add_norm,
                p,
                dict(D=d, BLOCK=triton.next_power_of_2(d), NEXT=nxt, BF=bf, EPS=EPS, SCALE=scale),
                {},
            )
        for ff in (3072, 2048, 512):
            yield mlp._gelu_up, dict(gp=dt, up=dt, out=dt, GS="i32", US="i32"), dict(FF=ff, BLOCK=1024, BF=bf), {}
    for g, ng in ((64, 12), (1024, 1)):
        for has_o in (False, True):
            p = dict(o=BF, r=BF, wp=F32, wn=F32, ro=BF, q=I8, f=F32, R="i32")
            yield I._add_quant, p, dict(D=768, DP=1024, BM=4, HAS_O=has_o, EPS=EPS, G=g, NG=ng), launch(I.TILES["rows"])
    yield (
        I._quant_rows,
        dict(x=BF, q=I8, f=F32, R="i32"),
        dict(D=768, DP=1024, BM=4, G=64, NG=12),
        launch(I.TILES["rows"]),
    )
    c = I.TILES["qkv"]
    p = dict(a=I8, f=F32, wa=I8, wb=I8, sa=F32, sb=F32, na=F32, nb=F32, cs=BF, out=BF, R="i32", L="i32")
    yield I._qkv_rope, p, dict(D=768, H=12, BM=c["BM"], HB=c["HB"], BK=c["BK"], EPS=EPS), launch(c)
    c = I.TILES["gate_up"]
    p = dict(a=I8, f=F32, w=I8, sw=F32, wu=I8, su=F32, c=I8, gs=F32, R="i32", N="i32", D="i32")
    yield I._gate_up, p, dict(BM=c["BM"], BN=c["BN"], BK=c["BK"]), launch(c)
    c = I.TILES["groups"]
    for d in (768, 3072):
        p = dict(a=I8, s=F32, w=I8, sw=F32, c=BF, R="i32", N="i32")
        yield I._gemm_groups, p, dict(D=d, BM=c["BM"], BN=c["BN"], G=64), launch(c)


@pytest.mark.parametrize(("fn", "types", "consts", "options"), list(configs()))
def test_compile(fn, types, consts, options):
    signature = {n: types.get(n, "constexpr") for n in fn.arg_names}
    source = triton.compiler.ASTSource(fn, signature, constexprs=consts)
    triton.compile(source, target=GPUTarget("cuda", 86, 32), options=dict(num_warps=4, num_stages=3) | options)
