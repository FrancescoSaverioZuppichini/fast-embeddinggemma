"""Fast EmbeddingGemma 2 embeddings: pictures and search queries into one 768-d space, on an NVIDIA card."""

import os
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import numpy as np

from .modeling import canvases, embed, load, queries
from .preprocessing import process

__version__ = version("fast-embeddinggemma")
__all__ = ["canvases", "embed", "embed_images", "embed_texts", "load", "process"]

# [Agent] Pictures prepared ahead of the card: enough for full batches per canvas, few enough that RAM stays flat (~0.9 MB each).
CHUNK = 256


def cut(vectors, dim):
    """Matryoshka: the first dim values, back to unit length (the model is trained for 768, 512, 256 and 128)."""
    if dim == vectors.shape[1]:
        return vectors  # already unit length: the engine output, untouched
    v = vectors[:, :dim]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def read(i, image):
    try:
        return process(image)
    except Exception as error:
        name = image if isinstance(image, str | os.PathLike) else type(image).__name__
        raise ValueError(f"picture {i} ({name}) could not be read: {error}") from error


def embed_images(model, images, dim=768, threads=8):
    """Pictures (paths, bytes, PIL images, uint8 RGB arrays) -> (n, dim) float32 unit vectors, in input order."""
    images, out = list(images), np.empty((len(images), 768), np.float32)
    with ThreadPoolExecutor(threads) as pool:

        def submit(k):
            return [pool.submit(read, i, images[i]) for i in range(k, min(k + CHUNK, len(images)))]

        # [Agent] The next chunk is prepared on the CPU threads while the card embeds this one.
        ahead = submit(0)
        for k in range(0, len(images), CHUNK):
            pictures, ahead = [f.result() for f in ahead], submit(k + CHUNK)
            groups = {}
            for i, (grid, patches) in enumerate(pictures, k):
                groups.setdefault(grid, []).append((i, patches[None]))
            for grid, rows in groups.items():
                out[[i for i, _ in rows]] = embed(model, grid, [p for _, p in rows]).cpu().numpy()
    return cut(out, dim)


def embed_texts(model, texts, dim=768):
    """Search queries -> (n, dim) float32 unit vectors in the pictures' space. The query graphs are built on the first call."""
    if "queries" not in model:
        model["queries"] = queries.load(model)
    return cut(queries.embed(model["queries"], list(texts)), dim)
