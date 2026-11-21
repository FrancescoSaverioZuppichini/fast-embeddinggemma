"""Search queries -> vectors through the engine's own text tower: the query prompt, the tokenizer, padding to a length bucket,
the embedding rows (kept in host memory: the table never goes on the card), the masked tower, the mean over real tokens. One
recorded CUDA graph per (bucket, queries), captured into the engine's memory pool. Queries over 512 tokens are cut at 512."""

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from . import weights
from .engine import capture_graph
from .text import CFG as TEXT
from .text import make_rope, run_tower

PROMPT = "task: search result | query: "
BUCKETS = (16, 32, 64, 128, 256, 512)
SIZES = (8, 4, 2, 1)  # queries per graph, largest first
ROWS = 1024  # most tokens one graph holds: a graph's memory grows with its rows (822 MiB at 4,096)


def sizes_for(L):
    return [B for B in SIZES if B * L <= ROWS]


def run_query(emb, mask, T):
    """Embedded tokens (B, L, 512) + real-token mask (B, L) -> (B, 768) fp32 unit vectors: mean over real tokens, unit length."""
    y, m = run_tower(emb, T, mask).float(), mask.unsqueeze(-1).float()
    return F.normalize((y * m).sum(1) / m.sum(1), dim=-1)


@torch.inference_mode()
def load(E, checkpoint=None):
    """The tokenizer (cutting at 512 tokens), the embedding table in host memory, the RoPE of every bucket, the graphs."""
    T, dev, dtype = E["T"], E["T"]["norm"].device, E["T"]["out"].dtype
    tokenizer = Tokenizer.from_file(hf_hub_download(weights.REPO, "tokenizer.json"))
    tokenizer.enable_truncation(max(BUCKETS))
    table = weights.open_file(checkpoint).get_tensor("language_model.embed_tokens.weight")
    T["rope"].update({L: make_rope(L, dev, dtype) for L in BUCKETS})
    graphs = {}
    for L in BUCKETS:
        for B in sizes_for(L):
            # [Agent] Random rows, not zeros: an all-zero input gives zero vectors and would fail the unit-length check.
            emb = torch.randn(B, L, TEXT["dim"], device=dev).to(dtype)
            mask = torch.ones(B, L, dtype=torch.bool, device=dev)
            graph, out = capture_graph(lambda: run_query(emb, mask, T), E["pool"])  # noqa: B023  called right here
            graph.replay()
            if not torch.allclose(out.norm(dim=-1), torch.ones(B, device=dev), atol=1e-3):
                raise RuntimeError(f"query graph {L} x {B}: the replay is not unit vectors")
            graphs[L, B] = emb, mask, graph, out
    return dict(tokenizer=tokenizer, table=table, scale=torch.tensor(TEXT["dim"] ** 0.5, dtype=dtype), graphs=graphs)


def run_group(Q, ids, mask):
    """Token ids and their mask (B, L) on the host, L a bucket -> (B, 768) vectors through the (L, B) graph."""
    # [Agent] HF scales the rows in bf16 (embedding * bf16(sqrt 512)); same rounding here, on the host.
    B, L = ids.shape
    slot, keys, graph, out = Q["graphs"][L, B]
    slot.copy_(Q["table"][ids] * Q["scale"])
    keys.copy_(mask)
    graph.replay()
    return out.clone()


@torch.inference_mode()
def embed(Q, texts):
    """Texts -> (n, 768) float32 unit vectors in input order; queries of one bucket share a graph."""
    seqs = [e.ids for e in Q["tokenizer"].encode_batch([PROMPT + t for t in texts])]
    out, groups = torch.empty(len(texts), TEXT["out"], device=Q["graphs"][16, 1][3].device), {}
    for i, s in enumerate(seqs):
        groups.setdefault(next(L for L in BUCKETS if L >= len(s)), []).append(i)
    for L, rows in groups.items():
        at = 0
        while at < len(rows):
            B = next(b for b in sizes_for(L) if b <= len(rows) - at)
            take = rows[at : at + B]
            lens = torch.tensor([len(seqs[i]) for i in take])
            ids = torch.zeros(B, L, dtype=torch.long)
            for r, i in enumerate(take):
                ids[r, : lens[r]] = torch.tensor(seqs[i])
            out[take] = run_group(Q, ids, torch.arange(L) < lens[:, None])
            at += B
    return out.cpu().numpy()
