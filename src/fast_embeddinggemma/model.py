"""The EmbeddingGemma 2 image path at 140 soft tokens on static canvases. uint8 patches (n, 9 rows cols, 768) of one canvas ->
16-layer vision tower -> 3x3 pooling + embedder -> text tower (text.py) -> (n, 768) fp32 unit vectors; bf16 (eager-exact ops)
or INT8 vision products (int8.py). One CUDA graph per (canvas, batch size), one memory pool, one uint8 buffer.
API: load(device, precision, checkpoint) -> engine; embed(engine, grid, pieces) -> vectors; canvases() -> the 7 grids."""

import math
from functools import partial

import torch
import torch.nn.functional as F

from . import kernels as K
from . import text as X

CFG = dict(
    repo="google/embeddinggemma-2",
    sizes=(32, 24, 16, 12, 8, 7, 6, 5, 4, 3, 2, 1),  # graph sizes, largest first
    canvases=((11, 11), (10, 13), (9, 14), (8, 15), (13, 10), (14, 9), (15, 8)),  # rows x cols at 140 tokens
    # the vision_config of the checkpoint
    dim=768,
    heads=12,
    head=64,
    layers=16,
    theta=100.0,
    patch=16,
    pool=3,
)


def find_file(name):  # local HF cache only, never the network
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(CFG["repo"], name)
    assert isinstance(path, str), f"{name} of {CFG['repo']} is not in the HF cache"
    return path


def load_weights(device, dtype, path):
    """Vision weights as stored; each layer through text.read_layer, the loader both towers share."""
    from safetensors import safe_open

    f = safe_open(path, "pt", device="cpu")
    mat, p = lambda k: f.get_tensor(k).to(device, dtype), "vision_tower.encoder.layers."
    return dict(
        patch=mat("vision_tower.patch_embedder.input_proj.weight"),
        proj=mat("embed_vision.embedding_projection.weight"),
        pos=mat("vision_tower.patch_embedder.position_embedding_table"),
        layers=[X.read_layer(f, f"{p}{i}.", "linear.weight", CFG["head"], dtype, device) for i in range(CFG["layers"])],
    )


def make_canvas(W, rows, cols):
    """Constants of one canvas, built once: the position-embedding sum per patch (HF's two lookups and add) and the
    RoPE table (2, L, head) = cos, sin in HF's [fx fx fy fy] layout, in the weights' dtype as HF rounds it."""
    dev, n, half = W["pos"].device, CFG["pool"], CFG["head"] // 2  # half: channels per axis
    y, x = torch.meshgrid(torch.arange(rows * n), torch.arange(cols * n), indexing="ij")
    x, y = x.flatten(), y.flatten()  # raster order, x fastest = the processor's
    # [Agent] HF builds inv_freq on the CPU at load time; building it on the card could differ in the last bit.
    inv = 1.0 / CFG["theta"] ** (torch.arange(0, half, 2, dtype=torch.float) / half)
    fx, fy = x[:, None].float() * inv, y[:, None].float() * inv
    ang = torch.cat([fx, fx, fy, fy], -1).to(dev)
    return dict(
        bias=W["pos"][0][x.to(dev)] + W["pos"][1][y.to(dev)],
        rows=rows,
        cols=cols,
        cs=torch.stack([ang.cos(), ang.sin()]).to(W["pos"].dtype),
    )


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
    a = K.attend(K.prep_qkv(F.linear(x, lw["qkv"]), cs, lw["qk"], H, H, D // 4), cs.shape[1], H, H, D)
    h, x = K.add_norm(h, F.linear(a, lw["wo"]), lw["w_pa"], lw["w_pre"])
    return K.add_norm(h, F.linear(K.gelu_up(*F.linear(x, lw["gu"]).chunk(2, -1)), lw["wd"]), lw["w_pf"], w_next)


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
    return F.linear(K.rms_norm(t.to(h.dtype)), W["proj"]).view(-1, rows * cols, W["proj"].shape[0])


def run_model(u8, E, cv):
    return X.run_text(pool_tokens(run_encoder(u8, E, cv), E["W"], cv), E["T"])


def prepare(device="cuda", precision="int8", checkpoint=None):
    """Weights + layer functions, no graphs; checkpoint: HF-layout safetensors. Default INT8 (R@10 -1.54 pt, 1.62x stock)."""
    if precision not in ("bf16", "int8", "fp32"):
        raise ValueError(f"precision is bf16, int8 or fp32, not {precision}")
    dtype, path = torch.float32 if precision == "fp32" else torch.bfloat16, checkpoint or find_file("model.safetensors")
    E = dict(
        W=load_weights(device, dtype, path),
        T=X.load_text(device, dtype, [r * c + 4 for r, c in CFG["canvases"]], path),
        start=K.rms_norm,
        layer=run_layer,
    )
    if precision == "int8":
        from . import int8

        # [Agent] The first pre-norm is add_quant with no branch to add: it hands the first QKV product INT8 rows + factors.
        E.update(
            W=int8.quantize(E["W"]), layer=int8.run_layer, start=lambda h, w: int8.add_quant(h, None, None, w, True)[1:]
        )
    return E


def capture_graph(fn, pool):
    """Run fn once on a side stream (Triton compiles, cuBLAS picks its kernels), then record it into the shared pool."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        out = fn()
    return graph, out


@torch.inference_mode()
def build_graphs(E):
    """Every (canvas, size) graph over one weight set and one pool (graphs never run at once, so a capture reuses what
    earlier captures freed); each reads a prefix of one uint8 buffer. A graph whose replay is not unit vectors raises."""
    pool, sizes, P, dev = torch.cuda.graph_pool_handle(), CFG["sizes"], 3 * CFG["patch"] ** 2, E["W"]["pos"].device
    buf = torch.zeros(max(sizes) * 9 * max(r * c for r, c in CFG["canvases"]) * P, dtype=torch.uint8, device=dev)
    graphs = {}
    for rows, cols in CFG["canvases"]:
        cv = E["tables"][rows, cols]
        for b in sizes:
            slot = buf[: b * 9 * rows * cols * P].view(b, -1, P)
            graph, out = capture_graph(partial(run_model, slot, E, cv), pool)
            graph.replay()
            if not torch.allclose(out.norm(dim=-1), torch.ones(b, device=dev), atol=1e-3):
                raise RuntimeError(f"graph {rows}x{cols} batch {b}: the replay is not unit vectors")
            graphs[(rows, cols), b] = slot, graph, out
    return graphs


def load(device="cuda", precision="int8", checkpoint=None):
    E = prepare(device, precision, checkpoint)
    # [Agent] Graphs hold raw pointers, not references: the tables must live with the engine or replays read freed memory.
    E["tables"] = {g: make_canvas(E["W"], *g) for g in CFG["canvases"]}
    E["graphs"] = build_graphs(E)
    # [Agent] Hand load-time temporaries back to the driver so reserved memory tracks what the engine keeps (2,702 -> 1,847 MiB).
    torch.cuda.empty_cache()
    return E


def split_batch(n):
    """n pictures -> graph sizes summing to n exactly (no padding), largest first: any n <= 32 takes at most two replays."""
    out = []
    while n:
        out.append(next(s for s in CFG["sizes"] if s <= n))
        n -= out[-1]
    return out


def fill_slot(slot, pieces, at):
    """Copy rows [at, at + len(slot)) of the pieces, taken end to end, straight into a graph's input slot."""
    for p in pieces:
        lo, hi = max(at, 0), min(at + len(slot), len(p))
        if lo < hi:
            slot[lo - at : hi - at].copy_(p[lo:hi], non_blocking=True)
        at -= len(p)


@torch.inference_mode()
def embed(E, grid, pieces):
    """pieces: uint8 (n, 9 rows cols, 768) of one canvas, or a list of such tensors (pinned RAM or card), copied straight into
    the graph slots (no torch.cat, no temporary buffer) -> (n, 768) fp32 unit vectors on the card."""
    pieces = [pieces] if torch.is_tensor(pieces) else pieces
    out, i = torch.empty(sum(len(p) for p in pieces), X.CFG["out"], device=E["W"]["pos"].device), 0
    for b in split_batch(len(out)):
        slot, graph, res = E["graphs"][grid, b]
        fill_slot(slot, pieces, i)
        graph.replay()
        out[i : i + b] = res
        i += b
    return out


def canvases():
    return CFG["canvases"]
