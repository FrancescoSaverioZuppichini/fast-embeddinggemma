"""The 7 canvases at 140 soft tokens, (rows, cols) of 3x3-patch cells: which one a picture goes on, and the constant tables
each one needs (position sums, 2-D RoPE), built once."""

import math

import torch

from .vision import CFG as VISION

CANVASES = ((11, 11), (10, 13), (9, 14), (8, 15), (13, 10), (14, 9), (15, 8))  # rows x cols at 140 tokens


def canvases():
    return CANVASES


def pick_canvas(width, height):
    # [Agent] Nearest in log aspect ratio, ties to more cells and then to the smaller (rows, cols): the locked native rule.
    r = math.log(width / height)
    return min(CANVASES, key=lambda g: (abs(math.log(g[1] / g[0]) - r), -g[0] * g[1], g))


def make_canvas(W, rows, cols):
    """Constants of one canvas, built once: the position-embedding sum per patch (HF's two lookups and add) and the
    RoPE table (2, L, head) = cos, sin in HF's [fx fx fy fy] layout, in the weights' dtype as HF rounds it."""
    dev, n, half = W["pos"].device, VISION["pool"], VISION["head"] // 2  # half: channels per axis
    y, x = torch.meshgrid(torch.arange(rows * n), torch.arange(cols * n), indexing="ij")
    x, y = x.flatten(), y.flatten()  # raster order, x fastest = the processor's
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
