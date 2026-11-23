"""Search quality on aspect-retrieval-5k (huggingface.co/datasets/Francesco/aspect-retrieval-5k): the gallery is the
thumbnail of every photo, the 5,040 test photos and the 20,000 distractors, and each test photo's three text queries,
its near-duplicate and its sketch must find it. R@1, R@5, R@10 and MRR per query kind, overall and per canvas, each with
a 95% bootstrap interval over the queries."""

import json
from pathlib import Path

import numpy as np

from . import embed_images, embed_texts, load

DATASET = "Francesco/aspect-retrieval-5k"
REVISION = "v1.0"
TEXTS = ("query_short", "query_sentence", "query_human")
PICTURES = ("neardup", "sketch")
# [Agent] The score each query kind is read by: R@1 where the model should name the photo outright, R@10 where a query
# can honestly fit several photos (a general human description, a line drawing).
HEADLINE = {"query_short": "R@1", "query_sentence": "R@1", "query_human": "R@10", "neardup": "R@1", "sketch": "R@10"}
RESAMPLES = 1000


def load_split(name: str, revision: str = REVISION, data: str | None = None):
    """`test` or `distractors` at a pinned revision, or from a copy saved with `save_to_disk` (what
    scripts/build_dataset.py writes). On the Hub each split is its own config of the same name: their columns differ."""
    import datasets

    if data:
        return datasets.load_from_disk(data)[name]
    return datasets.load_dataset(DATASET, name, revision=revision, split=name)


def get_ranks(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    # [Agent] Query i's photo is gallery row i (the test photos come first), so its rank is the number of photos that
    # score above it (0 = first).
    scores = queries @ gallery.T
    return (scores > scores[np.arange(len(queries)), np.arange(len(queries))][:, None]).sum(1)


def summarize(ranks: np.ndarray, rng: np.random.Generator) -> dict:
    """Each score with its 95% interval from resampling the queries."""
    hits = {"R@1": ranks < 1, "R@5": ranks < 5, "R@10": ranks < 10, "MRR": 1 / (ranks + 1)}
    picks = rng.integers(0, len(ranks), (RESAMPLES, len(ranks)))
    return {name: [float(v.mean()), *np.percentile(v[picks].mean(1), [2.5, 97.5]).tolist()] for name, v in hits.items()}


def print_report(report: dict) -> None:
    print("| queries | R@1 (95%) | R@5 | R@10 (95%) | MRR |\n|---|---|---|---|---|")
    for kind, scores in report.items():
        s = scores["all"]
        print(
            f"| {kind} | {s['R@1'][0]:.3f} ({s['R@1'][1]:.3f}-{s['R@1'][2]:.3f}) | {s['R@5'][0]:.3f} "
            f"| {s['R@10'][0]:.3f} ({s['R@10'][1]:.3f}-{s['R@10'][2]:.3f}) | {s['MRR'][0]:.3f} |"
        )
    canvases = [name for name in next(iter(report.values())) if name != "all"]
    print(f"\nheadline per canvas\n| queries | {' | '.join(canvases)} |\n|---|{'---|' * len(canvases)}")
    for kind, scores in report.items():
        cells = " | ".join(format(scores[c][HEADLINE[kind]][0], ".3f") for c in canvases)
        print(f"| {kind} {HEADLINE[kind]} | {cells} |")


def run(precision: str, dim: int, revision: str, data: str | None, out: str) -> None:
    test, distractors = load_split("test", revision, data), load_split("distractors", revision, data)
    model = load(precision)
    gallery = embed_images(model, list(test["thumb"]) + list(distractors["thumb"]), dim)
    vectors = {k: embed_texts(model, test[k], dim) for k in TEXTS} | {
        k: embed_images(model, test[k], dim) for k in PICTURES
    }
    canvases = np.array(list(test["canvas"]))
    rng = np.random.default_rng(0)
    report = {}
    for kind, queries in vectors.items():
        ranks = get_ranks(queries, gallery)
        report[kind] = {"all": summarize(ranks, rng)} | {
            c: summarize(ranks[canvases == c], rng) for c in dict.fromkeys(canvases)
        }
    print_report(report)
    setup = {"precision": precision, "dim": dim, "gallery": len(gallery), "revision": data or revision}
    Path(out).write_text(json.dumps({"setup": setup, "report": report}, indent=1))
    print(f"\n-> {out}")
