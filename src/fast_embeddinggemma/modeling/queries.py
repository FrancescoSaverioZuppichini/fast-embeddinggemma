"""Search queries -> vectors through the engine's own text tower: the query prompt, the tokenizer, padding to a length bucket,
the embedding rows (kept in host memory: the table never goes on the card), the masked tower, the mean over real tokens. One
recorded CUDA graph per (bucket, queries), captured into the engine's memory pool. A query over 512 tokens runs eager at its
exact length, with the sliding-window mask; past 8,192 tokens it is cut."""

from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from . import weights
from .engine import Engine, capture_graph
from .text import CFG as TEXT
from .text import make_rope, run_tower

PROMPT = "task: search result | query: "
BUCKETS = (16, 32, 64, 128, 256, 512)
# [Agent] Queries per graph, largest first; a graph holds at most ROWS tokens because its memory grows with them (822 MiB at
# 4,096 rows when captured alone).
SIZES = (8, 4, 2, 1)
ROWS = 1024
LIMIT = 8192


def fit_sizes(L: int) -> list[int]:
    return [B for B in SIZES if B * L <= ROWS]


def run_query(emb: torch.Tensor, mask: torch.Tensor, T: dict) -> torch.Tensor:
    """Embedded tokens (B, L, 512) + real-token mask (B, L) -> (B, 768) fp32 unit vectors: mean over real tokens, unit length."""
    y, m = run_tower(emb, T, mask).float(), mask.unsqueeze(-1).float()
    return F.normalize((y * m).sum(1) / m.sum(1), dim=-1)


@torch.inference_mode()
def load(E: Engine) -> dict:
    """The tokenizer (cutting at LIMIT tokens), the embedding table in host memory, the RoPE of every bucket, the graphs."""
    T = E["T"]
    tokenizer = Tokenizer.from_file(hf_hub_download(weights.REPO, "tokenizer.json"))
    tokenizer.enable_truncation(LIMIT)
    table = weights.open_file(E["checkpoint"]).get_tensor("language_model.embed_tokens.weight")
    T["rope"].update({L: make_rope(L) for L in BUCKETS})
    graphs = {}
    for L in BUCKETS:
        for B in fit_sizes(L):
            # [Agent] Random rows, not zeros: an all-zero input gives zero vectors and would fail the unit-length check.
            emb = torch.randn(B, L, TEXT["dim"], device="cuda").to(torch.bfloat16)
            mask = torch.ones(B, L, dtype=torch.bool, device="cuda")
            graph, out = capture_graph(partial(run_query, emb, mask, T), E["pool"])
            graph.replay()
            if not torch.allclose(out.norm(dim=-1), torch.ones(B, device="cuda"), atol=1e-3):
                raise RuntimeError(f"query graph {L} x {B}: the replay is not unit vectors")
            graphs[L, B] = emb, mask, graph, out
    scale = torch.tensor(TEXT["dim"] ** 0.5, dtype=torch.bfloat16)
    return dict(T=T, tokenizer=tokenizer, table=table, scale=scale, graphs=graphs)


def replay_bucket(Q: dict, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Token ids and their mask (B, L) on the host, L a bucket -> (B, 768) vectors through the (L, B) graph."""
    # [Agent] HF scales the rows in bf16 (embedding * bf16(sqrt 512)); same rounding here, on the host.
    B, L = ids.shape
    slot, keys, graph, out = Q["graphs"][L, B]
    slot.copy_(Q["table"][ids] * Q["scale"])
    keys.copy_(mask)
    graph.replay()
    return out.clone()


def embed_long(Q: dict, seq: list[int]) -> torch.Tensor:
    """One query over 512 tokens -> (768,) vector: eager at its exact length, with a RoPE table built for this call only."""
    L = len(seq)
    emb = (Q["table"][torch.tensor(seq)] * Q["scale"]).to("cuda")[None]
    mask = torch.ones(1, L, dtype=torch.bool, device="cuda")
    return run_query(emb, mask, Q["T"] | {"rope": {L: make_rope(L)}})[0]


@torch.inference_mode()
def embed(Q: dict, texts: list[str]) -> np.ndarray:
    """Texts -> (n, 768) float32 unit vectors in input order; queries of one bucket share a graph."""
    seqs = [e.ids for e in Q["tokenizer"].encode_batch([PROMPT + t for t in texts])]
    out = torch.empty(len(texts), TEXT["out"], device="cuda")
    groups = {}
    for i, s in enumerate(seqs):
        L = next((b for b in BUCKETS if b >= len(s)), None)
        if L is None:
            out[i] = embed_long(Q, s)
        else:
            groups.setdefault(L, []).append(i)
    for L, rows in groups.items():
        at = 0
        while at < len(rows):
            B = next(b for b in fit_sizes(L) if b <= len(rows) - at)
            take = rows[at : at + B]
            lens = torch.tensor([len(seqs[i]) for i in take])
            ids = torch.zeros(B, L, dtype=torch.long)
            for r, i in enumerate(take):
                ids[r, : lens[r]] = torch.tensor(seqs[i])
            out[take] = replay_bucket(Q, ids, torch.arange(L) < lens[:, None])
            at += B
    return out.cpu().numpy()
