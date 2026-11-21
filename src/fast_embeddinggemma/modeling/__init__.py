"""EmbeddingGemma 2 in PyTorch + Triton. Read vision.py, text.py, then model.py top to bottom; engine.py and queries.py run it
through CUDA graphs; convert.py makes our weights from Google's checkpoint; kernels/ holds the Triton kernels."""

from . import queries
from .canvas import CANVASES
from .engine import Engine, embed, load
from .model import EmbeddingGemma2

__all__ = ["CANVASES", "EmbeddingGemma2", "Engine", "embed", "load", "queries"]
