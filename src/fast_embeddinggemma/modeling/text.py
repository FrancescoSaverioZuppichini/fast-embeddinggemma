"""The text tower (EmbeddingGemma2TextModel, bidirectional): the vision layer family and kernels plus GQA, 1-D RoPE, a
per-layer-input (PLE) block and a layer scalar. [bos, boi, soft tokens, eoi, eos] -> 24 layers -> norm -> 512 -> 768 -> mean
-> unit length (the stock sidecar). No mask: a static canvas has no padding and every input is shorter than the 512-token window."""

import torch
import torch.nn.functional as F

from .attention import attend, prep_qkv
from .mlp import gelu_up
from .norm import add_norm, rms_norm

CFG = dict(
    dim=512,
    heads=4,
    layers=24,
    full_layers=(5, 11, 17, 23),
    window=512,
    out=768,
    sliding=dict(head=256, kv=2, theta=1e4),
    full=dict(head=512, kv=1, theta=1e6),
    special=(2, 255999, 258882, 1),
)  # bos, boi, eoi, eos token ids


def make_rope(L, device, dtype):
    """cos, sin tables (2, L, head) for positions 0..L-1, keyed by head size (256 sliding, 512 global), as HF builds them."""
    tables = {}
    for kind in (CFG["sliding"], CFG["full"]):
        hd = kind["head"]
        inv = 1.0 / kind["theta"] ** (torch.arange(0, hd, 2, dtype=torch.float) / hd)
        f = (torch.arange(L, dtype=torch.float)[:, None] * inv).to(device)
        tables[hd] = torch.stack([torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()]).to(dtype)
    return tables


def run_text_layer(h, x, lw, ple, rope, w_next):
    """Text layer on residual h (N, 512) and its input norm x -> h and the next input norm (w_next: next layer's or final)."""
    hd, H = lw["qk"].shape[1], CFG["heads"]
    qkv = prep_qkv(F.linear(x, lw["qkv"]), rope[hd], lw["qk"], H, lw["kv"], hd // 2)
    h, x = add_norm(h, F.linear(attend(qkv, rope[hd].shape[1], H, lw["kv"], hd), lw["wo"]), lw["w_pa"], lw["w_pre"])
    h, _ = add_norm(
        h, F.linear(gelu_up(*F.linear(x, lw["gu"]).chunk(2, -1)), lw["wd"]), lw["w_pf"], None
    )  # PLE reads raw h
    # [Agent] PLE block: gate gelu * slice and (h + norm) * scalar + the next input norm, each one kernel.
    return add_norm(h, F.linear(gelu_up(F.linear(h, lw["gate"]), ple), lw["proj"]), lw["w_ple"], w_next, lw["scalar"])


def run_text(tokens, T):
    """Soft tokens (B, n, 512) -> (B, 768) fp32 unit vectors."""
    B, n, D = tokens.shape
    s = T["special"].expand(B, 4, D)
    x = torch.cat([s[:, :2], tokens, s[:, 2:]], 1).reshape(B * (n + 4), D)
    # [Agent] PLE: one projection of the input to 24 x 512, scaled by 512^-0.5 and normed per layer; layer i reads slice i.
    ple = rms_norm((F.linear(x, T["ple"]) * D**-0.5).view(-1, CFG["layers"], D), T["ple_norm"])
    h, x, lws = x, rms_norm(x, T["layers"][0]["w_in"]), T["layers"]
    for i, lw in enumerate(lws):
        h, x = run_text_layer(
            h, x, lw, ple[:, i], T["rope"][n + 4], lws[i + 1]["w_in"] if i + 1 < len(lws) else T["norm"]
        )
    return F.normalize(F.linear(x, T["out"]).view(B, n + 4, -1).float().mean(1), dim=-1)
