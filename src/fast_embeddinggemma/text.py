"""The text tower (EmbeddingGemma2TextModel, bidirectional): the vision layer family and kernels plus GQA, 1-D RoPE, a
per-layer-input (PLE) block and a layer scalar. [bos, boi, soft tokens, eoi, eos] -> 24 layers -> norm -> 512 -> 768 -> mean
-> unit length (the stock sidecar). No mask: a static canvas has no padding and every input is shorter than the 512-token window."""

import torch
import torch.nn.functional as F

from . import kernels as K

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


def read_layer(f, p, w, head, dtype, device):
    """A sandwich-norm layer built on the CPU, moved once: q|k|v, gate|up concatenated; norms fp32 as HF; w = key ending."""
    mat, vec = lambda k: f.get_tensor(p + k).to(dtype), lambda k: f.get_tensor(p + k).float()
    d = dict(
        qkv=torch.cat([mat(f"self_attn.{n}_proj.{w}") for n in "qkv"]),
        gu=torch.cat([mat(f"mlp.{n}_proj.{w}") for n in ("gate", "up")]),
        wo=mat(f"self_attn.o_proj.{w}"),
        wd=mat(f"mlp.down_proj.{w}"),
        w_in=vec("input_layernorm.weight"),
        w_pre=vec("pre_feedforward_layernorm.weight"),
        w_pa=vec("post_attention_layernorm.weight"),
        w_pf=vec("post_feedforward_layernorm.weight"),
        qk=torch.stack([vec("self_attn.q_norm.weight"), vec("self_attn.k_norm.weight"), torch.ones(head)]),
    )
    return {k: v.to(device) for k, v in d.items()}


def load_text(device, dtype, lengths, path):
    """Text layers (read_layer + PLE block), special rows times the bf16 scale (sqrt(512) -> 22.625), RoPE of each length."""
    from safetensors import safe_open

    assert max(lengths) <= CFG["window"], "the sliding window would need a mask"
    f = safe_open(path, "pt", device="cpu")
    mat, vec = (
        lambda k: f.get_tensor("language_model." + k).to(device, dtype),
        lambda k: f.get_tensor("language_model." + k).to(device).float(),
    )

    def layer(i):
        p, kind = f"layers.{i}.", CFG["full" if i in CFG["full_layers"] else "sliding"]
        return read_layer(f, "language_model." + p, "weight", kind["head"], dtype, device) | dict(
            kv=kind["kv"],
            gate=mat(p + "ple_block.per_layer_input_gate.weight"),
            proj=mat(p + "ple_block.per_layer_projection.weight"),
            w_ple=vec(p + "ple_block.post_per_layer_input_norm.weight"),
            scalar=mat(p + "layer_scalar"),
        )

    rows = torch.cat([f.get_slice("language_model.embed_tokens.weight")[t : t + 1] for t in CFG["special"]]).to(
        device, dtype
    )
    return dict(
        layers=[layer(i) for i in range(CFG["layers"])],
        ple=mat("ple.per_layer_model_projection.weight"),
        ple_norm=vec("ple.per_layer_projection_norm.weight"),
        norm=vec("norm.weight"),
        out=mat("embedding_projection.weight"),
        special=rows * torch.tensor(CFG["dim"] ** 0.5, dtype=dtype),
        rope={L: make_rope(L, device, dtype) for L in lengths},
    )


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
    qkv = K.prep_qkv(F.linear(x, lw["qkv"]), rope[hd], lw["qk"], H, lw["kv"], hd // 2)
    h, x = K.add_norm(h, F.linear(K.attend(qkv, rope[hd].shape[1], H, lw["kv"], hd), lw["wo"]), lw["w_pa"], lw["w_pre"])
    h, _ = K.add_norm(
        h, F.linear(K.gelu_up(*F.linear(x, lw["gu"]).chunk(2, -1)), lw["wd"]), lw["w_pf"], None
    )  # PLE reads raw h
    # [Agent] PLE block: gate gelu * slice and (h + norm) * scalar + the next input norm, each one kernel.
    return K.add_norm(
        h, F.linear(K.gelu_up(F.linear(h, lw["gate"]), ple), lw["proj"]), lw["w_ple"], w_next, lw["scalar"]
    )


def run_text(tokens, T):
    """Soft tokens (B, n, 512) -> (B, 768) fp32 unit vectors."""
    B, n, D = tokens.shape
    s = T["special"].expand(B, 4, D)
    x = torch.cat([s[:, :2], tokens, s[:, 2:]], 1).reshape(B * (n + 4), D)
    # [Agent] PLE: one projection of the input to 24 x 512, scaled by 512^-0.5 and normed per layer; layer i reads slice i.
    ple = K.rms_norm((F.linear(x, T["ple"]) * D**-0.5).view(-1, CFG["layers"], D), T["ple_norm"])
    h, x, lws = x, K.rms_norm(x, T["layers"][0]["w_in"]), T["layers"]
    for i, lw in enumerate(lws):
        h, x = run_text_layer(
            h, x, lw, ple[:, i], T["rope"][n + 4], lws[i + 1]["w_in"] if i + 1 < len(lws) else T["norm"]
        )
    return F.normalize(F.linear(x, T["out"]).view(B, n + 4, -1).float().mean(1), dim=-1)
