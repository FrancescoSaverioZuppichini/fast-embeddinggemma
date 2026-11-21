"""The EmbeddingGemma 2 image model: canvases, weights, the vision and text towers with their kernels, the CUDA-graph engine."""

from .canvas import canvases
from .engine import embed, load

__all__ = ["canvases", "embed", "load"]
