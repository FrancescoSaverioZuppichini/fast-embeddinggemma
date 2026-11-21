"""The engine: the weights plus one CUDA graph per (canvas, batch size), over one memory pool and one uint8 input buffer.
load() builds it; embed() copies one canvas's uint8 patches into a graph's input and replays it -> (n, 768) unit vectors."""

from functools import partial

import torch

from . import int8, weights
from .canvas import CANVASES, make_canvas
from .norm import rms_norm
from .text import CFG as TEXT
from .text import make_rope, run_text
from .vision import CFG as VISION
from .vision import pool_tokens, run_encoder, run_layer

SIZES = (32, 24, 16, 12, 8, 7, 6, 5, 4, 3, 2, 1)  # graph sizes, largest first


def prepare(device="cuda", precision="int8", checkpoint=None):
    """Weights + layer functions, no graphs; checkpoint: HF-layout safetensors. Default INT8 (R@10 -1.54 pt, 1.62x stock)."""
    if precision not in ("bf16", "int8", "fp32"):
        raise ValueError(f"precision is bf16, int8 or fp32, not {precision}")
    dtype, f = torch.float32 if precision == "fp32" else torch.bfloat16, weights.open_file(checkpoint)
    lengths = [r * c + 4 for r, c in CANVASES]
    assert max(lengths) <= TEXT["window"], "the sliding window would need a mask"
    E = dict(
        W=weights.read_vision(f, device, dtype),
        T=weights.read_text(f, device, dtype) | dict(rope={L: make_rope(L, device, dtype) for L in lengths}),
        start=rms_norm,
        layer=run_layer,
    )
    if precision == "int8":
        # [Agent] The first pre-norm is add_quant with no branch to add: it hands the first QKV product INT8 rows + factors.
        E.update(
            W=int8.quantize(E["W"]), layer=int8.run_layer, start=lambda h, w: int8.add_quant(h, None, None, w, True)[1:]
        )
    return E


def run_model(u8, E, cv):
    return run_text(pool_tokens(run_encoder(u8, E, cv), E["W"], cv), E["T"])


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
    # [Agent] The pool is kept on the engine: the query graphs (queries.py) capture into it and add almost no memory.
    E["pool"] = pool = torch.cuda.graph_pool_handle()
    sizes, P, dev = SIZES, 3 * VISION["patch"] ** 2, E["W"]["pos"].device
    buf = torch.zeros(max(sizes) * 9 * max(r * c for r, c in CANVASES) * P, dtype=torch.uint8, device=dev)
    graphs = {}
    for rows, cols in CANVASES:
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
    E["tables"] = {g: make_canvas(E["W"], *g) for g in CANVASES}
    E["graphs"] = build_graphs(E)
    # [Agent] Hand load-time temporaries back to the driver so reserved memory tracks what the engine keeps (2,702 -> 1,847 MiB).
    torch.cuda.empty_cache()
    return E


def split_batch(n):
    """n pictures -> graph sizes summing to n exactly (no padding), largest first: any n <= 32 takes at most two replays."""
    out = []
    while n:
        out.append(next(s for s in SIZES if s <= n))
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
    out, i = torch.empty(sum(len(p) for p in pieces), TEXT["out"], device=E["W"]["pos"].device), 0
    for b in split_batch(len(out)):
        slot, graph, res = E["graphs"][grid, b]
        fill_slot(slot, pieces, i)
        graph.replay()
        out[i : i + b] = res
        i += b
    return out
