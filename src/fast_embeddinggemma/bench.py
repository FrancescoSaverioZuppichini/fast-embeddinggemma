"""The quick benchmark: prepare the pictures (outside the clock, on CPU threads; that rate is printed too), put them on the card,
then embed them in calls of up to b pictures per canvas: 2 warm-up sweeps + 5 timed sweeps per size, img/s from the median sweep.
Model-only on purpose: the card's speed, not the disk's or the decoder's."""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .modeling import Engine, embed, load
from .modeling.canvas import Grid
from .preprocessing import process

WARMUPS, TIMED = 2, 5


def make_calls(pictures: list[tuple[Grid, torch.Tensor]], b: int) -> list[tuple[Grid, torch.Tensor]]:
    """Pictures grouped by canvas in input order, cut into calls of up to b, each one block on the card."""
    groups = {}
    for grid, patches in pictures:
        groups.setdefault(grid, []).append(patches)
    return [(g, torch.stack(ps[k : k + b]).cuda()) for g, ps in groups.items() for k in range(0, len(ps), b)]


def time_sweep(E: Engine, calls: list[tuple[Grid, torch.Tensor]]) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for grid, block in calls:
        embed(E, grid, block)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run(paths: list[Path], precision: str, sizes: tuple[int, ...], threads: int) -> None:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(threads) as pool:
        pictures = list(pool.map(process, paths))
    print(
        f"prep: {len(pictures)} pictures on {threads} threads, {len(pictures) / (time.perf_counter() - t0):.0f} img/s"
    )
    t0 = time.perf_counter()
    E = load(precision, max_batch=max(32, *sizes))
    print(f"load: {time.perf_counter() - t0:.1f} s ({precision}, {len(E['graphs'])} graphs)", flush=True)
    for b in sizes:
        calls = make_calls(pictures, b)
        for _ in range(WARMUPS):
            time_sweep(E, calls)
        rates = sorted(len(pictures) / time_sweep(E, calls) for _ in range(TIMED))
        print(
            f"{precision} batch {b}: {statistics.median(rates):.1f} img/s (min {rates[0]:.1f}, max {rates[-1]:.1f})",
            flush=True,
        )
