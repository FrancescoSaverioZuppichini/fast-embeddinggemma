"""The vision tower, top to bottom: the config, VisionLayer (bf16), Int8VisionLayer (the default: the same layer with INT8
products), then VisionTower: uint8 patches of one canvas -> 16 layers -> 3x3 average -> image tokens of 512 numbers, the
input the text tower reads in place of words (HF calls them soft tokens). The Triton kernels it calls are in kernels/."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from .kernels import int8
from .kernels.attention import attend, qkv_rope
from .kernels.mlp import gelu_up
from .kernels.norm import add_norm, rms_norm


@dataclass(frozen=True)
class Config:
    """The vision_config of the checkpoint. An image token is pool x pool patches of patch x patch pixels: 48 x 48."""

    dim: int = 768
    heads: int = 12
    head: int = 64
    layers: int = 16
    ff: int = 3072
    theta: float = 100.0
    patch: int = 16
    pool: int = 3
    positions: int = 10240
    out: int = 512


CFG = Config()


class VisionLayer(nn.Module):
    """norm -> attention (q/k norms, 2-D RoPE) -> norm + add, norm -> GELU MLP -> norm + add. Each add_norm kernel adds a
    branch to the residual and also returns the next norm, so a layer hands the next one its input already normed."""

    def __init__(self) -> None:
        super().__init__()
        d, heads, head, ff = CFG.dim, CFG.heads, CFG.head, CFG.ff
        self.qkv_proj = nn.Linear(d, 3 * heads * head, bias=False)  # q|k|v in one product
        self.o_proj = nn.Linear(heads * head, d, bias=False)
        self.gate_up_proj = nn.Linear(d, 2 * ff, bias=False)  # gate|up in one product
        self.down_proj = nn.Linear(ff, d, bias=False)
        self.qk_norm = nn.Parameter(torch.empty(3, head))  # q norm, k norm, ones for v
        self.input_layernorm = nn.Parameter(torch.empty(d))
        self.post_attention_layernorm = nn.Parameter(torch.empty(d))
        self.pre_feedforward_layernorm = nn.Parameter(torch.empty(d))
        self.post_feedforward_layernorm = nn.Parameter(torch.empty(d))

    def pre_norm(self, h: torch.Tensor) -> torch.Tensor:
        """The input norm of the first layer; every later layer gets it from the previous layer's last add_norm."""
        return rms_norm(h, self.input_layernorm)

    def forward(
        self, h: torch.Tensor, x: torch.Tensor, rope: torch.Tensor, next_norm: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Residual h (N, 768) and its input norm x -> h and the next layer's input (next_norm: its weight, None at the end)."""
        heads, head = CFG.heads, CFG.head
        # [Agent] 2-D RoPE: each axis owns 32 of the 64 channels and pairs channel c with c + 16 inside them (half = head / 4).
        qkv = qkv_rope(self.qkv_proj(x), rope, self.qk_norm, heads, heads, head // 4)
        a = attend(qkv, rope.shape[1], heads, heads, head)
        h, x = add_norm(h, self.o_proj(a), self.post_attention_layernorm, self.pre_feedforward_layernorm)
        y = self.down_proj(gelu_up(*self.gate_up_proj(x).chunk(2, -1)))
        return add_norm(h, y, self.post_feedforward_layernorm, next_norm)


class Int8VisionLayer(nn.Module):
    """VisionLayer with INT8 products: weights per channel, activations one scale per 64 values (the MLP input: per row).
    The QKV product runs q/k norms + RoPE in its epilogue, gate|up runs GELU and INT8 packing. Built from a loaded layer;
    the INT8 tensors are plain attributes because they are made on the card at load and never saved."""

    def __init__(self, layer: VisionLayer) -> None:
        super().__init__()
        gate, up = layer.gate_up_proj.weight.chunk(2)
        self.qkv = int8.split_heads(*int8.quantize_weight(layer.qkv_proj.weight), layer.qk_norm)
        self.o = int8.quantize_weight(layer.o_proj.weight)
        self.gate_up = int8.quantize_weight(gate) + int8.quantize_weight(up)
        self.down = int8.quantize_weight(layer.down_proj.weight)
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.pre_feedforward_layernorm = layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm = layer.post_feedforward_layernorm

    def pre_norm(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [Agent] add_quant with nothing to add: the first QKV product gets INT8 rows + their factors, never a bf16 norm.
        return int8.add_quant(h, None, None, self.input_layernorm, grouped=True)[1:]

    def forward(
        self, h: torch.Tensor, x: tuple[torch.Tensor, torch.Tensor], rope: torch.Tensor, next_norm: torch.Tensor | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """VisionLayer.forward; x = (INT8 rows, 64-value group factors) of the input norm, and so is what it returns."""
        heads = h.shape[1] // 64
        a = attend(int8.qkv_rope(*x, self.qkv, rope), rope.shape[1], heads, heads, 64)
        # [Agent] The MLP input gets one scale per row (grouped False): its product is the fused gate|up kernel.
        o = int8.gemm_groups(*int8.quant_rows(a), *self.o)
        h, q, f = int8.add_quant(h, o, self.post_attention_layernorm, self.pre_feedforward_layernorm, grouped=False)
        # [Agent] The last layer has no next norm; its INT8 rows are never read, so any norm weight fills the slot.
        y = int8.gemm_groups(*int8.gate_up(q, f, *self.gate_up), *self.down)
        nxt = self.post_feedforward_layernorm if next_norm is None else next_norm
        h, q, f = int8.add_quant(h, y, self.post_feedforward_layernorm, nxt, grouped=True)
        return h, (q, f)


class VisionTower(nn.Module):
    """uint8 patches of one canvas -> patch product + position sums -> 16 layers -> 3x3 average -> image tokens (B, n, 512)."""

    def __init__(self) -> None:
        super().__init__()
        d = CFG.dim
        self.patch_proj = nn.Linear(3 * CFG.patch**2, d, bias=False)
        self.position_table = nn.Parameter(torch.empty(2, CFG.positions, d))  # x and y lookups, summed per patch
        self.layers = nn.ModuleList(VisionLayer() for _ in range(CFG.layers))
        self.out_proj = nn.Linear(d, CFG.out, bias=False)

    def quantize(self) -> None:
        """Swap every layer for its INT8 twin (the patch product, positions and out projection stay bf16)."""
        self.layers = nn.ModuleList(Int8VisionLayer(layer) for layer in self.layers)

    def forward(self, patches: torch.Tensor, canvas: dict) -> torch.Tensor:
        """uint8 patches (B, 9 rows cols, 768) -> image tokens (B, rows cols, 512); canvas: make_canvas's tables."""
        # [Agent] Pixels are cast to bf16 after dividing by 255 and before 2 (p - 0.5), as the reference does; HF fed fp32
        # pixels rounds after the subtraction instead (127 of 256 byte values differ).
        pixels = 2 * ((patches.float() * (1 / 255)).to(self.patch_proj.weight.dtype) - 0.5)
        h = (self.patch_proj(pixels) + canvas["bias"]).view(-1, CFG.dim)
        x = self.layers[0].pre_norm(h)
        for i, layer in enumerate(self.layers):
            nxt = self.layers[i + 1].input_layernorm if i + 1 < len(self.layers) else None
            h, x = layer(h, x, canvas["cs"], nxt)
        rows, cols, n = canvas["rows"], canvas["cols"], CFG.pool
        # [Agent] Raster order makes the 3x3 average a reshape-mean. It stays fp32 through the sqrt(768) scale, as the
        # reference does; HF's own pooler rounds it to bf16 once more.
        t = h.view(-1, rows, n, cols, n, CFG.dim).float().mean((2, 4)) * math.sqrt(CFG.dim)
        return self.out_proj(rms_norm(t.to(h.dtype))).view(-1, rows * cols, CFG.out)
