"""The MLP middle of both towers: gelu_tanh(gate) * up in one kernel."""

import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh

from .norm import round_to


@triton.jit
def _gelu_up(gp, up, out, GS, US, FF: tl.constexpr, BLOCK: tl.constexpr, BF: tl.constexpr):
    row, c = tl.program_id(0), tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(gp + row * GS + c, c < FF, 0.0).to(tl.float32)
    u = tl.load(up + row * US + c, c < FF, 0.0).to(tl.float32)
    # [Agent] torch's own gelu_tanh term by term (same constants, same order, same libdevice tanh), so the bf16 result is
    # eager's; an exp form was as accurate but differed from eager in 0.1-7% of entries.
    act = round_to(0.5 * g * (1.0 + tanh(0.7978845608028654 * (g + 0.044715 * (g * g * g)))), BF)
    tl.store(out + row * FF + c, round_to(act * u, BF).to(out.dtype.element_ty), c < FF)


def gelu_up(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """gelu_tanh(g) * u for two (N, FF) row-strided views (the halves of a merged gate|up product, or a PLE gate and slice)."""
    N, FF = g.shape
    out = torch.empty(N, FF, dtype=g.dtype, device=g.device)
    _gelu_up[(N, triton.cdiv(FF, 1024))](
        g, u, out, g.stride(0), u.stride(0), FF=FF, BLOCK=1024, BF=g.dtype == torch.bfloat16
    )
    return out
