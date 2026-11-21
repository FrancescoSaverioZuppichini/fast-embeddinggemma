"""Google's checkpoint -> our weights, once: the vision and text towers under model.py's names, q|k|v and gate|up fused, norms
in fp32 as HF keeps them, the special rows scaled, the token table for queries, and the tokenizer beside them. Run by
`fast-embeddinggemma convert OUT_DIR`; load() downloads the published copy of OUT_DIR (engine.REPO) by default. This is the
only file that knows the checkpoint's names."""

import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

from .text import CFG as TEXT
from .vision import CFG as VISION

GOOGLE = "google/embeddinggemma-2"


def read_layer(f: safe_open, src: str, dst: str, suffix: str, head: int) -> dict[str, torch.Tensor]:
    """One HF layer at src -> our layer's tensors at dst; suffix is how that tower names its weight keys."""

    def mat(key: str) -> torch.Tensor:
        return f.get_tensor(src + key).to(torch.bfloat16)

    def vec(key: str) -> torch.Tensor:
        return f.get_tensor(src + key).float()

    return {
        dst + "qkv_proj.weight": torch.cat([mat(f"self_attn.{n}_proj.{suffix}") for n in "qkv"]),
        dst + "o_proj.weight": mat(f"self_attn.o_proj.{suffix}"),
        dst + "gate_up_proj.weight": torch.cat([mat(f"mlp.{n}_proj.{suffix}") for n in ("gate", "up")]),
        dst + "down_proj.weight": mat(f"mlp.down_proj.{suffix}"),
        dst + "qk_norm": torch.stack(
            [vec("self_attn.q_norm.weight"), vec("self_attn.k_norm.weight"), torch.ones(head)]
        ),
        dst + "input_layernorm": vec("input_layernorm.weight"),
        dst + "post_attention_layernorm": vec("post_attention_layernorm.weight"),
        dst + "pre_feedforward_layernorm": vec("pre_feedforward_layernorm.weight"),
        dst + "post_feedforward_layernorm": vec("post_feedforward_layernorm.weight"),
    }


def read_text_layer(f: safe_open, i: int) -> dict[str, torch.Tensor]:
    src, dst = f"language_model.layers.{i}.", f"text.layers.{i}."
    kind = TEXT.full if i in TEXT.full_layers else TEXT.sliding
    return read_layer(f, src, dst, "weight", kind.head) | {
        dst + "ple_gate.weight": f.get_tensor(src + "ple_block.per_layer_input_gate.weight").to(torch.bfloat16),
        dst + "ple_proj.weight": f.get_tensor(src + "ple_block.per_layer_projection.weight").to(torch.bfloat16),
        dst + "ple_norm": f.get_tensor(src + "ple_block.post_per_layer_input_norm.weight").float(),
        dst + "layer_scalar": f.get_tensor(src + "layer_scalar").to(torch.bfloat16),
    }


def convert(out_dir: str, checkpoint: str | None = None) -> Path:
    """Google's model.safetensors (or a local file in its layout) -> OUT_DIR/model.safetensors + tokenizer.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = safe_open(checkpoint or hf_hub_download(GOOGLE, "model.safetensors"), "pt", device="cpu")
    state = {
        "vision.patch_proj.weight": f.get_tensor("vision_tower.patch_embedder.input_proj.weight").to(torch.bfloat16),
        "vision.position_table": f.get_tensor("vision_tower.patch_embedder.position_embedding_table").to(
            torch.bfloat16
        ),
        "vision.out_proj.weight": f.get_tensor("embed_vision.embedding_projection.weight").to(torch.bfloat16),
        "text.ple_proj.weight": f.get_tensor("language_model.ple.per_layer_model_projection.weight").to(torch.bfloat16),
        "text.ple_norm": f.get_tensor("language_model.ple.per_layer_projection_norm.weight").float(),
        "text.norm": f.get_tensor("language_model.norm.weight").float(),
        "text.out_proj.weight": f.get_tensor("language_model.embedding_projection.weight").to(torch.bfloat16),
        "embed_tokens": f.get_tensor("language_model.embed_tokens.weight"),
    }
    for i in range(VISION.layers):
        state |= read_layer(f, f"vision_tower.encoder.layers.{i}.", f"vision.layers.{i}.", "linear.weight", VISION.head)
    for i in range(TEXT.layers):
        state |= read_text_layer(f, i)
    # [Agent] Scaled on the card in bf16, the way the engine always did it, so the rows are bit for bit the same.
    rows = torch.cat([f.get_slice("language_model.embed_tokens.weight")[t : t + 1] for t in TEXT.special])
    state["special"] = (rows.to("cuda", torch.bfloat16) * torch.tensor(TEXT.dim**0.5, dtype=torch.bfloat16)).cpu()
    save_file({k: v.contiguous() for k, v in state.items()}, out / "model.safetensors", metadata={"source": GOOGLE})
    shutil.copy(hf_hub_download(GOOGLE, "tokenizer.json"), out / "tokenizer.json")
    return out
