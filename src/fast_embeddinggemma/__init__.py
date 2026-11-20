"""EmbeddingGemma 2 image embeddings at 140 soft tokens: PyTorch + Triton, CUDA graphs, INT8 by default."""

from importlib.metadata import version

import numpy as np

from .model import canvases, embed, load
from .prep import read

__version__ = version("fast-embeddinggemma")
__all__ = ["canvases", "embed", "embed_images", "load", "read"]


def embed_images(E, images):
    """Paths or bytes -> (n, 768) float32 unit vectors in input order; each canvas group is one engine call."""
    pictures = [read(x) for x in images]
    groups = {}
    for i, (grid, _) in enumerate(pictures):
        groups.setdefault(grid, []).append(i)
    out = np.empty((len(pictures), 768), np.float32)
    for grid, rows in groups.items():
        out[rows] = embed(E, grid, [pictures[i][1][None] for i in rows]).cpu().numpy()
    return out
