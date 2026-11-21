"""The text tower, top to bottom: the config, the RoPE tables, TextLayer, then TextTower: embedded tokens -> 24 bidirectional
layers -> norm -> 768 numbers per token. Pictures reach it as [bos, boi, image tokens, eoi, eos] (model.py); search queries
as embedded words with a padding mask. Same kernel family as the vision tower, plus GQA, 1-D RoPE, a per-layer input (PLE)
and a layer scalar."""

from dataclasses import dataclass

import torch
from torch import nn

from .kernels.attention import attend, qkv_rope
from .kernels.mlp import gelu_up
from .kernels.norm import add_norm, rms_norm


@dataclass(frozen=True)
class Attention:
    head: int
    kv: int
    theta: float


@dataclass(frozen=True)
class Config:
    """The text_config of the checkpoint: 20 sliding-window layers (head 256, 2 kv heads), 4 global ones (head 512, 1 kv)."""

    dim: int = 512
    heads: int = 4
    layers: int = 24
    full_layers: tuple[int, ...] = (5, 11, 17, 23)
    window: int = 512
    ff: int = 2048
    out: int = 768
    sliding: Attention = Attention(head=256, kv=2, theta=1e4)
    full: Attention = Attention(head=512, kv=1, theta=1e6)
    special: tuple[int, ...] = (2, 255999, 258882, 1)  # bos, boi, eoi, eos token ids


CFG = Config()


def make_rope(L: int) -> dict[int, torch.Tensor]:
    """cos, sin tables (2, L, head) for positions 0..L-1, keyed by head size (256 sliding, 512 global), as HF builds them."""
    tables = {}
    for kind in (CFG.sliding, CFG.full):
        hd = kind.head
        inv = 1.0 / kind.theta ** (torch.arange(0, hd, 2, dtype=torch.float) / hd)
        f = (torch.arange(L, dtype=torch.float)[:, None] * inv).to("cuda")
        tables[hd] = torch.stack([torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()]).to(torch.bfloat16)
    return tables


class TextLayer(nn.Module):
    """norm -> attention (GQA, q/k norms, 1-D RoPE) -> norm + add, norm -> GELU MLP -> norm + add, then the PLE block:
    gate(h) * this layer's slice of the per-layer input -> projection -> norm + add, times the layer scalar."""

    def __init__(self, full: bool) -> None:
        super().__init__()
        kind = CFG.full if full else CFG.sliding
        d, heads, ff = CFG.dim, CFG.heads, CFG.ff
        self.head, self.kv = kind.head, kind.kv
        self.qkv_proj = nn.Linear(d, (heads + 2 * self.kv) * self.head, bias=False)  # q|k|v in one product
        self.o_proj = nn.Linear(heads * self.head, d, bias=False)
        self.gate_up_proj = nn.Linear(d, 2 * ff, bias=False)  # gate|up in one product
        self.down_proj = nn.Linear(ff, d, bias=False)
        self.qk_norm = nn.Parameter(torch.empty(3, self.head))  # q norm, k norm, ones for v
        self.input_layernorm = nn.Parameter(torch.empty(d))
        self.post_attention_layernorm = nn.Parameter(torch.empty(d))
        self.pre_feedforward_layernorm = nn.Parameter(torch.empty(d))
        self.post_feedforward_layernorm = nn.Parameter(torch.empty(d))
        self.ple_gate = nn.Linear(d, d, bias=False)
        self.ple_proj = nn.Linear(d, d, bias=False)
        self.ple_norm = nn.Parameter(torch.empty(d))
        self.layer_scalar = nn.Parameter(torch.empty(1))

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        ple: torch.Tensor,
        rope: dict[int, torch.Tensor],
        next_norm: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Residual h (N, 512), its input norm x, this layer's PLE slice -> h and the next input norm (next_norm: its weight)."""
        heads, head, cs = CFG.heads, self.head, rope[self.head]
        qkv = qkv_rope(self.qkv_proj(x), cs, self.qk_norm, heads, self.kv, head // 2)
        a = attend(qkv, cs.shape[1], heads, self.kv, head, mask)
        h, x = add_norm(h, self.o_proj(a), self.post_attention_layernorm, self.pre_feedforward_layernorm)
        # [Agent] No next norm here: the PLE block reads the raw residual.
        y = self.down_proj(gelu_up(*self.gate_up_proj(x).chunk(2, -1)))
        h, _ = add_norm(h, y, self.post_feedforward_layernorm, None)
        # [Agent] PLE block: gate gelu * slice, then (h + norm) * scalar + the next input norm, each one kernel.
        y = self.ple_proj(gelu_up(self.ple_gate(h), ple))
        return add_norm(h, y, self.ple_norm, next_norm, self.layer_scalar)


class TextTower(nn.Module):
    """Embedded tokens (B, L, 512) -> PLE projection -> 24 layers -> norm -> 768 numbers per token (B, L, 768)."""

    def __init__(self) -> None:
        super().__init__()
        d = CFG.dim
        self.ple_proj = nn.Linear(d, CFG.layers * d, bias=False)
        self.ple_norm = nn.Parameter(torch.empty(d))
        self.layers = nn.ModuleList(TextLayer(i in CFG.full_layers) for i in range(CFG.layers))
        self.norm = nn.Parameter(torch.empty(d))
        self.out_proj = nn.Linear(d, CFG.out, bias=False)
        # [Agent] RoPE tables by sequence length, made before any CUDA graph capture (a capture cannot copy from the host)
        # and kept here because a recorded graph reads them by address.
        self.rope: dict[int, dict[int, torch.Tensor]] = {}

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, rope: dict[int, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Embedded tokens (B, L, 512) -> (B, L, 768) before the mean; mask (B, L) bool = real tokens; rope: tables for L
        when they are not kept (a one-off long query)."""
        B, L, D = x.shape
        rope = self.rope[L] if rope is None else rope
        x = x.reshape(B * L, D)
        band = None
        if L > CFG.window:
            # [Agent] Past 512 tokens the sliding layers see a key only within 512 positions (HF: |q - kv| <= 512, real keys).
            i = torch.arange(L, device=x.device)
            band = (i[:, None] - i[None, :]).abs() <= CFG.window
            band = band[None] if mask is None else mask[:, None, :] & band
        # [Agent] PLE: one projection of the input to 24 x 512, scaled by 512^-0.5 and normed per layer; layer i reads slice i.
        ple = rms_norm((self.ple_proj(x) * D**-0.5).view(-1, CFG.layers, D), self.ple_norm)
        h, x = x, rms_norm(x, self.layers[0].input_layernorm)
        for i, layer in enumerate(self.layers):
            nxt = self.layers[i + 1].input_layernorm if i + 1 < len(self.layers) else self.norm
            layer_mask = band if band is not None and i not in CFG.full_layers else mask
            h, x = layer(h, x, ple[:, i], rope, nxt, layer_mask)
        return self.out_proj(x).view(B, L, -1)
