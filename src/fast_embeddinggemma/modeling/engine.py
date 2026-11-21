"""Pictures through recorded CUDA graphs: the model plus one graph per (canvas, batch size), over one memory pool and one uint8
input buffer. load() downloads our converted weights (convert.py) on first use and builds it; embed() copies one canvas's uint8
patches into a graph's input and replays it -> (n, 768) unit vectors."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal

import torch
from huggingface_hub import snapshot_download

from .canvas import CANVASES, Grid, make_canvas
from .model import EmbeddingGemma2, load_model
from .text import CFG as TEXT
from .text import make_rope
from .vision import CFG as VISION

# [Agent] Our converted weights and tokenizer, made once by convert.py from google/embeddinggemma-2 and uploaded.
REPO = "Francesco/fast-embeddinggemma"
# [Agent] Graph sizes, largest first. Past 32 pictures a call gains under 1% on an RTX 3090 at 300 W (the card is power-bound)
# and costs about 1.4 GB of card memory per doubling, so 32 is the default ceiling; load(max_batch=...) records bigger ones.
ALL_SIZES = (128, 96, 64, 48, 32, 24, 16, 12, 8, 7, 6, 5, 4, 3, 2, 1)

Precision = Literal["int8", "bf16"]


@dataclass
class Engine:
    model: EmbeddingGemma2
    weights: str  # the converted file; queries read the token table and the tokenizer beside it
    sizes: tuple[int, ...]
    # [Agent] Graphs hold raw pointers, not references: the canvas tables must live as long as the engine.
    canvases: dict[Grid, dict]
    pool: Any = None
    graphs: dict = field(default_factory=dict)
    queries: dict | None = None  # queries.load(engine), on the first search query


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
    E.pool = torch.cuda.graph_pool_handle()
    P = 3 * VISION.patch**2
    buf = torch.zeros(max(E.sizes) * 9 * max(r * c for r, c in CANVASES) * P, dtype=torch.uint8, device="cuda")
    graphs = {}
    for rows, cols in CANVASES:
        for b in E.sizes:
            slot = buf[: b * 9 * rows * cols * P].view(b, -1, P)
            graph, out = capture_graph(partial(E.model.embed_images, slot, E.canvases[rows, cols]), E.pool)
            graph.replay()
            if not torch.allclose(out.norm(dim=-1), torch.ones(b, device="cuda"), atol=1e-3):
                raise RuntimeError(f"graph {rows}x{cols} batch {b}: the replay is not unit vectors")
            graphs[(rows, cols), b] = slot, graph, out
    return graphs


def load(precision: Precision = "int8", weights: str | None = None, max_batch: int = 32) -> Engine:
    """The engine, ready: the model, canvas tables and one graph per (canvas, size) up to max_batch pictures a call.
    weights: a converted file (`fast-embeddinggemma convert`); by default our published one, downloaded once."""
    if precision not in ("int8", "bf16"):
        raise ValueError(f"precision is int8 or bf16, not {precision}")
    # [Agent] Both files in one snapshot folder: queries read the tokenizer from beside the weights.
    path = weights or str(
        Path(snapshot_download(REPO, allow_patterns=["model.safetensors", "tokenizer.json"])) / "model.safetensors"
    )
    model = load_model(path, precision)
    # [Agent] A picture is [bos, boi, rows x cols image tokens, eoi, eos]: one RoPE table per canvas length.
    model.text.rope.update({r * c + 4: make_rope(r * c + 4) for r, c in CANVASES})
    canvases = {g: make_canvas(model.vision.position_table, *g) for g in CANVASES}
    E = Engine(model, path, tuple(s for s in ALL_SIZES if s <= max_batch), canvases)
    E.graphs = build_graphs(E)
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
    out, i = torch.empty(sum(len(p) for p in pieces), TEXT.out, device="cuda"), 0
    for b in split_batch(len(out), E.sizes):
        slot, graph, res = E.graphs[grid, b]
        fill_slot(slot, pieces, i)
        graph.replay()
        out[i : i + b] = res
        i += b
    return out
