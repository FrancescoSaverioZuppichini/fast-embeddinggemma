"""The vision tower: uint8 patches -> [-1, 1] -> patch product + position sums -> 16 sandwich-norm layers (q/k norms, 2-D
RoPE, GELU MLP) -> 3x3 average pooling + embedder -> soft tokens of 512. This is the bf16 layer; int8.run_layer is the INT8 one."""

import math

import torch.nn.functional as F

from .attention import attend, prep_qkv
from .mlp import gelu_up
from .norm import add_norm, rms_norm

CFG = dict(dim=768, heads=12, head=64, layers=16, theta=100.0, patch=16, pool=3)  # the vision_config of the checkpoint


def scale_pixels(u8, dtype):
    # [Agent] The stock sidecar casts pixel_values (u8 / 255) to bf16 BEFORE the model and HF then computes 2 (p - 0.5) in
    # bf16; this follows it. HF fed fp32 pixels rounds after the subtraction instead (127 of 256 byte values differ).
    return 2 * ((u8.float() * (1 / 255)).to(dtype) - 0.5)


def embed_patches(u8, W, cv):
    """uint8 patches (B, L, 768) -> first hidden state (B, L, 768): pixels to [-1, 1], patch product, position sums."""
    return F.linear(scale_pixels(u8, W["patch"].dtype), W["patch"]) + cv["bias"]


def run_layer(h, x, lw, w_next, cs):
    """One sandwich-norm layer on the flat residual h (N, 768); x = its pre-norm, the input of the next product."""
    H, D = CFG["heads"], CFG["head"]
    # [Agent] 2-D RoPE: each axis owns 32 of the 64 channels and pairs channel c with c + 16 inside them (half = head / 4).
    a = attend(prep_qkv(F.linear(x, lw["qkv"]), cs, lw["qk"], H, H, D // 4), cs.shape[1], H, H, D)
    h, x = add_norm(h, F.linear(a, lw["wo"]), lw["w_pa"], lw["w_pre"])
    return add_norm(h, F.linear(gelu_up(*F.linear(x, lw["gu"]).chunk(2, -1)), lw["wd"]), lw["w_pf"], w_next)


def run_encoder(u8, E, cv):
    """uint8 patches (B, L, 768) of one canvas -> last hidden state, flat (B L, 768)."""
    W, lws = E["W"], E["W"]["layers"]
    h = embed_patches(u8, W, cv).view(-1, CFG["dim"])
    x = E["start"](h, lws[0]["w_in"])
    for i, lw in enumerate(lws):
        h, x = E["layer"](h, x, lw, lws[i + 1]["w_in"] if i + 1 < len(lws) else None, cv["cs"])
    return h


def pool_tokens(h, W, cv):
    """Last hidden state -> soft tokens (B, rows cols, 512) for the text tower."""
    rows, cols, n = cv["rows"], cv["cols"], CFG["pool"]
    # [Agent] Raster order makes the 3x3 pooling a reshape-mean. It stays fp32 through the sqrt(768) scale like the stock
    # sidecar; HF's own pooler rounds the mean to bf16 once more, so this follows the sidecar, not HF.
    t = h.view(-1, rows, n, cols, n, CFG["dim"]).float().mean((2, 4)) * math.sqrt(CFG["dim"])
    return F.linear(rms_norm(t.to(h.dtype)), W["proj"]).view(-1, rows * cols, W["proj"].shape[0])
