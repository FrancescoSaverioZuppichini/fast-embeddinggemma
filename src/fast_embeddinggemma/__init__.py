"""Fast EmbeddingGemma 2 embeddings: pictures and search queries into one 768-d space, on an NVIDIA card."""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import numpy as np

from .modeling import CANVASES, Engine, embed, load, queries
from .preprocessing import Picture, process

__version__ = version("fast-embeddinggemma")
__all__ = ["CANVASES", "Engine", "Picture", "embed", "embed_images", "embed_texts", "load", "process"]

# [Agent] Pictures prepared ahead of the card: enough for full batches per canvas, few enough that RAM stays flat (~0.9 MB each).
CHUNK = 256


def cut(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka: the first dim values, back to unit length (the model is trained for 768, 512, 256 and 128)."""
    if dim == vectors.shape[1]:
        return vectors
    v = vectors[:, :dim]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def embed_images(model: Engine, images: Iterable[Picture], dim: int = 768, threads: int = 8) -> np.ndarray:
    """Pictures (paths, bytes, PIL images, uint8 RGB arrays) -> (n, dim) float32 unit vectors, in input order. A picture that
    cannot be read raises ValueError naming its position."""
    images = list(images)
    out = np.empty((len(images), 768), np.float32)
    with ThreadPoolExecutor(threads) as pool:
        # [Agent] The next chunk is prepared on the CPU threads while the card embeds this one.
        ahead = [pool.submit(process, x) for x in images[:CHUNK]]
        for k in range(0, len(images), CHUNK):
            done, ahead = ahead, [pool.submit(process, x) for x in images[k + CHUNK : k + 2 * CHUNK]]
            groups = {}
            for i, future in enumerate(done, k):
                try:
                    grid, patches = future.result()
                except Exception as error:
                    raise ValueError(f"picture {i} could not be read: {error}") from error
                groups.setdefault(grid, []).append((i, patches[None]))
            for grid, rows in groups.items():
                out[[i for i, _ in rows]] = embed(model, grid, [p for _, p in rows]).cpu().numpy()
    return cut(out, dim)


def embed_texts(model: Engine, texts: Iterable[str], dim: int = 768) -> np.ndarray:
    """Search queries -> (n, dim) float32 unit vectors in the pictures' space. The query graphs are built on the first call."""
    if "queries" not in model:
        model["queries"] = queries.load(model)
    return cut(queries.embed(model["queries"], list(texts)), dim)
