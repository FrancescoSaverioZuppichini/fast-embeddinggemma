"""The EmbeddingGemma 2 image model: canvases, weights, the vision and text towers with their kernels, the CUDA-graph engine."""

from . import queries
from .canvas import CANVASES
from .engine import Engine, embed, load

__all__ = ["CANVASES", "Engine", "embed", "load", "queries"]
