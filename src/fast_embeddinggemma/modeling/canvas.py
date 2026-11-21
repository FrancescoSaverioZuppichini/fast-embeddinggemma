"""The 7 canvases at 140 soft tokens, (rows, cols) of 3x3-patch cells: which one a picture goes on, and the constant tables
each one needs (position sums, 2-D RoPE), built once."""

import math

import torch

from .vision import CFG as VISION

Grid = tuple[int, int]
CANVASES: tuple[Grid, ...] = ((11, 11), (10, 13), (9, 14), (8, 15), (13, 10), (14, 9), (15, 8))


def pick_canvas(width: int, height: int) -> Grid:
    # [Agent] Nearest in log aspect ratio; ties go to more cells, then to the smaller (rows, cols).
    r = math.log(width / height)
    return min(CANVASES, key=lambda g: (abs(math.log(g[1] / g[0]) - r), -g[0] * g[1], g))


def make_canvas(W: dict, rows: int, cols: int) -> dict:
    """Constants of one canvas: the position-embedding sum per patch (HF's two lookups and add) and the RoPE table (2, L, head) =
    cos, sin in HF's [fx fx fy fy] layout, rounded to the weights' dtype as HF rounds it."""
    dev, n = W["pos"].device, VISION["pool"]
    # [Agent] Each axis owns half of a head's channels.
    half = VISION["head"] // 2
    y, x = torch.meshgrid(torch.arange(rows * n), torch.arange(cols * n), indexing="ij")
    # [Agent] Raster order, x fastest: the patch order of the preparation.
    x, y = x.flatten(), y.flatten()
    # [Agent] HF builds inv_freq on the CPU at load time; building it on the card could differ in the last bit.
    inv = 1.0 / VISION["theta"] ** (torch.arange(0, half, 2, dtype=torch.float) / half)
    fx, fy = x[:, None].float() * inv, y[:, None].float() * inv
    ang = torch.cat([fx, fx, fy, fy], -1).to(dev)
    return dict(
        bias=W["pos"][0][x.to(dev)] + W["pos"][1][y.to(dev)],
        rows=rows,
        cols=cols,
        cs=torch.stack([ang.cos(), ang.sin()]).to(W["pos"].dtype),
    )
