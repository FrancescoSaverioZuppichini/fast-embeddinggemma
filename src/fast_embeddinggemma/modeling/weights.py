"""The checkpoint: google/embeddinggemma-2's safetensors (downloaded once into the Hugging Face cache) or a local file in the
same layout, read into the layer dicts of the two towers. Every layer, vision or text, goes through read_layer."""

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from .text import CFG as TEXT
from .vision import CFG as VISION

REPO = "google/embeddinggemma-2"


def open_file(checkpoint=None):
    """A local HF-layout safetensors file, or the hub checkpoint (downloaded on first use; HF_HUB_OFFLINE=1 stays offline)."""
    return safe_open(checkpoint or hf_hub_download(REPO, "model.safetensors"), "pt", device="cpu")


def read_layer(f, p, w, head, dtype, device):
    """A sandwich-norm layer built on the CPU, moved once: q|k|v, gate|up concatenated; norms fp32 as HF; w = key ending."""
    mat, vec = lambda k: f.get_tensor(p + k).to(dtype), lambda k: f.get_tensor(p + k).float()
    d = dict(
        qkv=torch.cat([mat(f"self_attn.{n}_proj.{w}") for n in "qkv"]),
        gu=torch.cat([mat(f"mlp.{n}_proj.{w}") for n in ("gate", "up")]),
        wo=mat(f"self_attn.o_proj.{w}"),
        wd=mat(f"mlp.down_proj.{w}"),
        w_in=vec("input_layernorm.weight"),
        w_pre=vec("pre_feedforward_layernorm.weight"),
        w_pa=vec("post_attention_layernorm.weight"),
        w_pf=vec("post_feedforward_layernorm.weight"),
        qk=torch.stack([vec("self_attn.q_norm.weight"), vec("self_attn.k_norm.weight"), torch.ones(head)]),
    )
    return {k: v.to(device) for k, v in d.items()}


def read_vision(f, device, dtype):
    """Vision weights as stored: the patch product, the position tables, the embedder and the 16 layers."""
    mat, p = lambda k: f.get_tensor(k).to(device, dtype), "vision_tower.encoder.layers."
    return dict(
        patch=mat("vision_tower.patch_embedder.input_proj.weight"),
        proj=mat("embed_vision.embedding_projection.weight"),
        pos=mat("vision_tower.patch_embedder.position_embedding_table"),
        layers=[
            read_layer(f, f"{p}{i}.", "linear.weight", VISION["head"], dtype, device) for i in range(VISION["layers"])
        ],
    )


def read_text(f, device, dtype):
    """Text layers (read_layer + PLE block) and the 4 special rows times the bf16 scale (sqrt(512) -> 22.625)."""
    mat, vec = (
        lambda k: f.get_tensor("language_model." + k).to(device, dtype),
        lambda k: f.get_tensor("language_model." + k).to(device).float(),
    )

    def layer(i):
        p, kind = f"layers.{i}.", TEXT["full" if i in TEXT["full_layers"] else "sliding"]
        return read_layer(f, "language_model." + p, "weight", kind["head"], dtype, device) | dict(
            kv=kind["kv"],
            gate=mat(p + "ple_block.per_layer_input_gate.weight"),
            proj=mat(p + "ple_block.per_layer_projection.weight"),
            w_ple=vec(p + "ple_block.post_per_layer_input_norm.weight"),
            scalar=mat(p + "layer_scalar"),
        )

    rows = torch.cat([f.get_slice("language_model.embed_tokens.weight")[t : t + 1] for t in TEXT["special"]])
    return dict(
        layers=[layer(i) for i in range(TEXT["layers"])],
        ple=mat("ple.per_layer_model_projection.weight"),
        ple_norm=vec("ple.per_layer_projection_norm.weight"),
        norm=vec("norm.weight"),
        out=mat("embedding_projection.weight"),
        special=rows.to(device, dtype) * torch.tensor(TEXT["dim"] ** 0.5, dtype=dtype),
    )
