"""The checkpoint: google/embeddinggemma-2's safetensors (downloaded once into the Hugging Face cache) or a local file in the
same layout, read onto the card in bf16 as the layer dicts of the two towers. Every layer, vision or text, goes through read_layer."""

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from .text import CFG as TEXT
from .vision import CFG as VISION

REPO = "google/embeddinggemma-2"


def open_file(checkpoint: str | None = None) -> safe_open:
    """A local HF-layout safetensors file, or the hub checkpoint (downloaded on first use; HF_HUB_OFFLINE=1 stays offline)."""
    return safe_open(checkpoint or hf_hub_download(REPO, "model.safetensors"), "pt", device="cpu")


def read_layer(file: safe_open, prefix: str, suffix: str, head: int) -> dict[str, torch.Tensor]:
    """One sandwich-norm layer, built on the CPU and moved once: q|k|v and gate|up concatenated, norms in fp32 as HF keeps them.
    suffix is how this tower names its weight keys."""

    def mat(key: str) -> torch.Tensor:
        return file.get_tensor(prefix + key).to(torch.bfloat16)

    def vec(key: str) -> torch.Tensor:
        return file.get_tensor(prefix + key).float()

    layer = dict(
        qkv=torch.cat([mat(f"self_attn.{n}_proj.{suffix}") for n in "qkv"]),
        gu=torch.cat([mat(f"mlp.{n}_proj.{suffix}") for n in ("gate", "up")]),
        wo=mat(f"self_attn.o_proj.{suffix}"),
        wd=mat(f"mlp.down_proj.{suffix}"),
        w_in=vec("input_layernorm.weight"),
        w_pre=vec("pre_feedforward_layernorm.weight"),
        w_pa=vec("post_attention_layernorm.weight"),
        w_pf=vec("post_feedforward_layernorm.weight"),
        qk=torch.stack([vec("self_attn.q_norm.weight"), vec("self_attn.k_norm.weight"), torch.ones(head)]),
    )
    return {k: v.to("cuda") for k, v in layer.items()}


def read_vision(file: safe_open) -> dict:
    """Vision weights as stored: the patch product, the position tables, the embedder and the 16 layers."""

    def mat(key: str) -> torch.Tensor:
        return file.get_tensor(key).to("cuda", torch.bfloat16)

    prefix = "vision_tower.encoder.layers."
    return dict(
        patch=mat("vision_tower.patch_embedder.input_proj.weight"),
        proj=mat("embed_vision.embedding_projection.weight"),
        pos=mat("vision_tower.patch_embedder.position_embedding_table"),
        layers=[read_layer(file, f"{prefix}{i}.", "linear.weight", VISION["head"]) for i in range(VISION["layers"])],
    )


def read_text(file: safe_open) -> dict:
    """Text layers (read_layer + PLE block) and the 4 special rows times the bf16 scale (sqrt(512) -> 22.625)."""

    def mat(key: str) -> torch.Tensor:
        return file.get_tensor("language_model." + key).to("cuda", torch.bfloat16)

    def vec(key: str) -> torch.Tensor:
        return file.get_tensor("language_model." + key).to("cuda").float()

    def layer(i: int) -> dict:
        prefix, kind = f"layers.{i}.", TEXT["full" if i in TEXT["full_layers"] else "sliding"]
        return read_layer(file, "language_model." + prefix, "weight", kind["head"]) | dict(
            kv=kind["kv"],
            gate=mat(prefix + "ple_block.per_layer_input_gate.weight"),
            proj=mat(prefix + "ple_block.per_layer_projection.weight"),
            w_ple=vec(prefix + "ple_block.post_per_layer_input_norm.weight"),
            scalar=mat(prefix + "layer_scalar"),
        )

    rows = torch.cat([file.get_slice("language_model.embed_tokens.weight")[t : t + 1] for t in TEXT["special"]])
    return dict(
        layers=[layer(i) for i in range(TEXT["layers"])],
        ple=mat("ple.per_layer_model_projection.weight"),
        ple_norm=vec("ple.per_layer_projection_norm.weight"),
        norm=vec("norm.weight"),
        out=mat("embedding_projection.weight"),
        special=rows.to("cuda", torch.bfloat16) * torch.tensor(TEXT["dim"] ** 0.5, dtype=torch.bfloat16),
    )
