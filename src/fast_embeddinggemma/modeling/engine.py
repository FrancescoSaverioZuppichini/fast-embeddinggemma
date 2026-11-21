"""The engine: the weights plus one CUDA graph per (canvas, batch size), over one memory pool and one uint8 input buffer.
load() builds it; embed() copies one canvas's uint8 patches into a graph's input and replays it -> (n, 768) unit vectors."""

from collections.abc import Sequence
from functools import partial
from typing import Any, Literal

import torch

from . import int8, weights
from .canvas import CANVASES, Grid, make_canvas
from .norm import rms_norm
from .text import CFG as TEXT
from .text import encode, make_rope
from .vision import CFG as VISION
from .vision import pool_tokens, run_encoder, run_layer

# [Agent] Graph sizes, largest first. Past 32 pictures a call gains under 1% on an RTX 3090 at 300 W (the card is power-bound)
# and costs about 1.4 GB of card memory per doubling, so 32 is the default ceiling; load(max_batch=...) records bigger ones.
ALL_SIZES = (128, 96, 64, 48, 32, 24, 16, 12, 8, 7, 6, 5, 4, 3, 2, 1)

Precision = Literal["int8", "bf16"]
# [Agent] One plain dict: W, T, start, layer, checkpoint, sizes, tables, pool, graphs (+ queries once text is embedded).
Engine = dict[str, Any]


def prepare(precision: Precision = "int8", checkpoint: str | None = None) -> Engine:
    """Weights + layer functions, no graphs; checkpoint: an HF-layout safetensors file (default: the hub checkpoint)."""
    if precision not in ("int8", "bf16"):
        raise ValueError(f"precision is int8 or bf16, not {precision}")
    file = weights.open_file(checkpoint)
    rope = {L: make_rope(L) for L in (r * c + 4 for r, c in CANVASES)}
    E: Engine = dict(
        W=weights.read_vision(file),
        T=weights.read_text(file) | {"rope": rope},
        start=rms_norm,
        layer=run_layer,
        checkpoint=checkpoint,
    )
    if precision == "int8":
        # [Agent] The first pre-norm is add_quant with no branch to add: it hands the first QKV product INT8 rows + factors.
        E.update(
            W=int8.quantize(E["W"]),
            layer=int8.run_layer,
            start=lambda h, w: int8.add_quant(h, None, None, w, grouped=True)[1:],
        )
    return E


def run_model(u8: torch.Tensor, E: Engine, cv: dict) -> torch.Tensor:
    return encode(pool_tokens(run_encoder(u8, E, cv), E["W"], cv), E["T"])


def capture_graph(fn: partial, pool: Any) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
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
def build_graphs(E: Engine) -> dict:
    """Every (canvas, size) graph over one weight set and one pool (graphs never run at once, so a capture reuses what
    earlier captures freed); each reads a prefix of one uint8 buffer. A graph whose replay is not unit vectors raises."""
    # [Agent] The pool is kept on the engine: the query graphs (queries.py) capture into it and add almost no memory.
    E["pool"] = pool = torch.cuda.graph_pool_handle()
    P = 3 * VISION["patch"] ** 2
    buf = torch.zeros(max(E["sizes"]) * 9 * max(r * c for r, c in CANVASES) * P, dtype=torch.uint8, device="cuda")
    graphs = {}
    for rows, cols in CANVASES:
        cv = E["tables"][rows, cols]
        for b in E["sizes"]:
            slot = buf[: b * 9 * rows * cols * P].view(b, -1, P)
            graph, out = capture_graph(partial(run_model, slot, E, cv), pool)
            graph.replay()
            if not torch.allclose(out.norm(dim=-1), torch.ones(b, device="cuda"), atol=1e-3):
                raise RuntimeError(f"graph {rows}x{cols} batch {b}: the replay is not unit vectors")
            graphs[(rows, cols), b] = slot, graph, out
    return graphs


def load(precision: Precision = "int8", checkpoint: str | None = None, max_batch: int = 32) -> Engine:
    """The engine, ready: weights, canvas tables and one graph per (canvas, size) up to max_batch pictures a call."""
    E = prepare(precision, checkpoint)
    E["sizes"] = tuple(s for s in ALL_SIZES if s <= max_batch)
    # [Agent] Graphs hold raw pointers, not references: the tables must live with the engine or replays read freed memory.
    E["tables"] = {g: make_canvas(E["W"], *g) for g in CANVASES}
    E["graphs"] = build_graphs(E)
    # [Agent] Hand load-time temporaries back to the driver so reserved memory tracks what the engine keeps.
    torch.cuda.empty_cache()
    return E


def split_batch(n: int, sizes: Sequence[int]) -> list[int]:
    """n pictures -> graph sizes summing to n exactly (no padding), largest first: up to max_batch takes at most two replays."""
    out = []
    while n:
        out.append(next(s for s in sizes if s <= n))
        n -= out[-1]
    return out


def fill_slot(slot: torch.Tensor, pieces: list[torch.Tensor], at: int) -> None:
    """Copy rows [at, at + len(slot)) of the pieces, taken end to end, straight into a graph's input slot."""
    for p in pieces:
        lo, hi = max(at, 0), min(at + len(slot), len(p))
        if lo < hi:
            slot[lo - at : hi - at].copy_(p[lo:hi], non_blocking=True)
        at -= len(p)


@torch.inference_mode()
def embed(E: Engine, grid: Grid, pieces: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """pieces: uint8 (n, 9 rows cols, 768) of one canvas, or a list of such tensors (pinned RAM or card), copied straight into
    the graph slots (no torch.cat, no temporary buffer) -> (n, 768) fp32 unit vectors on the card."""
    pieces = [pieces] if torch.is_tensor(pieces) else pieces
    out, i = torch.empty(sum(len(p) for p in pieces), TEXT["out"], device="cuda"), 0
    for b in split_batch(len(out), E["sizes"]):
        slot, graph, res = E["graphs"][grid, b]
        fill_slot(slot, pieces, i)
        graph.replay()
        out[i : i + b] = res
        i += b
    return out
