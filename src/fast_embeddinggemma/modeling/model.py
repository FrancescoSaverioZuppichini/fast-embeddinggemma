"""EmbeddingGemma 2, put together: the vision tower turns a picture into image tokens, the text tower reads them like a
sentence and the mean of its outputs is the picture's vector; a search query goes through the same text tower. Both land in
one 768-d space of unit vectors, so cosine similarity is a dot product. load_model() reads our converted weights (convert.py)."""

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from .text import CFG as TEXT
from .text import TextTower
from .vision import VisionTower


class EmbeddingGemma2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision = VisionTower()
        self.text = TextTower()
        # [Agent] The bos, boi, eoi, eos rows of the token table, already times the bf16 embedding scale (sqrt(512) -> 22.625).
        self.special = nn.Parameter(torch.empty(4, TEXT.dim))

    def embed_images(self, patches: torch.Tensor, canvas: dict) -> torch.Tensor:
        """uint8 patches (B, L, 768) of one canvas -> (B, 768) fp32 unit vectors: [bos, boi, image tokens, eoi, eos] through
        the text tower, mean over every token, unit length."""
        tokens = self.vision(patches, canvas)
        s = self.special.expand(tokens.shape[0], 4, tokens.shape[2])
        x = torch.cat([s[:, :2], tokens, s[:, 2:]], 1)
        return F.normalize(self.text(x).float().mean(1), dim=-1)

    def embed_queries(
        self, emb: torch.Tensor, mask: torch.Tensor, rope: dict[int, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Embedded query tokens (B, L, 512) + real-token mask (B, L) -> (B, 768) fp32 unit vectors: mean over real tokens."""
        y, m = self.text(emb, mask, rope).float(), mask.unsqueeze(-1).float()
        return F.normalize((y * m).sum(1) / m.sum(1), dim=-1)


def load_model(path: str, precision: str = "int8") -> EmbeddingGemma2:
    """Our converted safetensors file -> the model on the card, frozen; INT8 swaps the vision layers for their INT8 twins.
    The token table in the file stays out: queries keep it in host memory."""
    # [Agent] Built on the meta device, so no memory is spent on weights that load_state_dict(assign=True) replaces anyway.
    with torch.device("meta"):
        model = EmbeddingGemma2()
    with safe_open(path, "pt", device="cuda") as f:
        state = {k: f.get_tensor(k) for k in f.keys() if k != "embed_tokens"}
    model.load_state_dict(state, assign=True)
    model.requires_grad_(False)
    if precision == "int8":
        model.vision.quantize()
    return model
