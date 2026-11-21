"""The quick benchmark: prepare the first n pictures of a folder (outside the clock, on CPU threads; that rate is printed too),
put them on the card, then embed them in calls of up to b pictures per canvas: 2 warm-up sweeps + 5 timed sweeps per size,
img/s from the median sweep. Model-only on purpose: the card's speed, not the disk's or the decoder's."""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from . import preprocessing
from .modeling import embed, load

CFG = dict(warmups=2, timed=5)


def prep_folder(folder, n, threads):
    """The first n files of the folder, by name -> [(grid, patches)] and the seconds it took."""
    paths = sorted(p for p in Path(folder).iterdir() if p.is_file())[:n]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(threads) as pool:
        pictures = list(pool.map(preprocessing.process, paths))
    return pictures, time.perf_counter() - t0


def make_calls(pictures, b):
    """Pictures grouped by canvas in folder order, cut into calls of up to b, each one block on the card."""
    groups = {}
    for grid, patches in pictures:
        groups.setdefault(grid, []).append(patches)
    return [(g, torch.stack(ps[k : k + b]).cuda()) for g, ps in groups.items() for k in range(0, len(ps), b)]


def time_sweep(E, calls):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for grid, block in calls:
        embed(E, grid, block)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run(folder, precision="int8", n=256, sizes=(1, 8, 32), threads=8):
    pictures, secs = prep_folder(folder, n, threads)
    print(f"prep: {len(pictures)} pictures on {threads} threads, {len(pictures) / secs:.0f} img/s", flush=True)
    t0 = time.perf_counter()
    E = load("cuda", precision)
    print(f"load: {time.perf_counter() - t0:.1f} s ({precision}, {len(E['graphs'])} graphs)", flush=True)
    for b in sizes:
        calls = make_calls(pictures, b)
        for _ in range(CFG["warmups"]):
            time_sweep(E, calls)
        rates = sorted(len(pictures) / time_sweep(E, calls) for _ in range(CFG["timed"]))
        print(
            f"{precision} batch {b}: {statistics.median(rates):.1f} img/s (min {rates[0]:.1f}, max {rates[-1]:.1f})",
            flush=True,
        )
