"""One GPU thread serves every request, HTTP or gRPC: search queries first, then the canvas whose oldest picture has waited
longest, up to max_batch pictures from any requests in one engine call. Pictures are prepared on CPU threads and each joins
the queue as soon as it is ready. The queue lives only in memory and only for in-flight work: durable queueing belongs to
the caller (stash keeps it in its database). Past max_queued pictures in flight a request gets Busy, which the caller
answers as 503 / RESOURCE_EXHAUSTED."""

import asyncio
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Awaitable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from opentelemetry import metrics, trace
from opentelemetry.metrics import Observation

from .. import cut
from ..modeling import CANVASES, embed, load, queries
from ..modeling.canvas import Grid
from ..preprocessing import process
from .config import Settings

log = logging.getLogger("fast_embeddinggemma")
tracer = trace.get_tracer("fast_embeddinggemma")
meter = metrics.get_meter("fast_embeddinggemma")
PICTURES = meter.create_counter("embed.pictures", unit="1", description="pictures embedded")
TEXTS = meter.create_counter("embed.texts", unit="1", description="search queries embedded")
FAILURES = meter.create_counter("embed.failures", unit="1", description="inputs that could not be embedded")
BATCH_MS = meter.create_histogram("embed.batch.duration", unit="ms", description="one engine call")

Item = tuple[str, str, str | bytes]  # (id, "path" | "data" | "text", source)


class Busy(Exception):
    """More pictures are in flight than EMBED_MAX_QUEUED, or the card is short of memory: try again later."""


def start(settings: Settings) -> dict:
    """Load the engine and its query graphs and start the GPU thread; every request works through the returned state."""
    engine = load(settings.precision, max_batch=settings.max_batch)
    engine.queries = queries.load(engine)
    state: dict[str, Any] = dict(
        engine=engine,
        settings=settings,
        prep=ThreadPoolExecutor(settings.threads),
        pictures={grid: deque() for grid in CANVASES},
        texts=deque(),
        ready=threading.Condition(),
        running=True,
        inflight=0,
    )
    meter.create_observable_gauge(
        "embed.queue.depth",
        callbacks=[lambda options: [Observation(state["inflight"])]],
        description="pictures in flight",
    )
    state["thread"] = threading.Thread(target=run, args=(state,), name="gpu", daemon=True)
    state["thread"].start()
    return state


def stop(state: dict) -> None:
    """Finish what was admitted: the CPU threads queue their pictures, then the GPU thread empties the queue and exits."""
    state["prep"].shutdown(cancel_futures=True)
    with state["ready"]:
        state["running"] = False
        state["ready"].notify()
    state["thread"].join()


def queued(state: dict) -> int:
    return sum(len(q) for q in state["pictures"].values())


def free_mb() -> int:
    return torch.cuda.mem_get_info()[0] >> 20


def health(state: dict) -> dict:
    settings, free = state["settings"], free_mb()
    # [Agent] A stopped GPU thread answers nothing, so it reads as not ok and callers pause instead of waiting on it.
    ok = free >= settings.min_free_mb and state["thread"].is_alive()
    return dict(
        model=settings.tag,
        dims=settings.dim,
        freeVramMb=free,
        queue=dict(images=state["inflight"], texts=len(state["texts"])),
        vram="ok" if ok else "low",
    )


def take(state: dict) -> tuple[Grid | None, list]:
    """Under the lock: all waiting queries, else up to max_batch pictures of the canvas whose oldest picture waited longest."""
    pictures, texts = state["pictures"], state["texts"]
    if texts:
        return None, [texts.popleft() for _ in range(len(texts))]
    grid = min((g for g, q in pictures.items() if q), key=lambda g: pictures[g][0][0])
    return grid, [pictures[grid].popleft() for _ in range(min(len(pictures[grid]), state["settings"].max_batch))]


def run(state: dict) -> None:
    """The GPU thread: waits for work, runs one batch outside the lock, hands every job its vector or the batch's error.
    Exits once stopped and the queue is empty, so shutdown finishes the work already admitted."""
    while True:
        with state["ready"]:
            while state["running"] and not state["texts"] and not queued(state):
                state["ready"].wait()
            if not state["texts"] and not queued(state):
                return
            grid, jobs = take(state)
        # [Agent] A job whose caller gave up (a cancelled request) is dropped before it costs card time; the rest are
        # marked running, so they can no longer be cancelled and setting their result cannot fail.
        jobs = [job for job in jobs if job[-1].set_running_or_notify_cancel()]
        if not jobs:
            continue
        try:
            values = run_texts(state, jobs) if grid is None else run_pictures(state, grid, jobs)
        except Exception as error:
            log.exception("embed.queries.failed" if grid is None else "embed.batch.failed", extra={"jobs": len(jobs)})
            values = [error] * len(jobs)
        for job, value in zip(jobs, values, strict=True):
            if isinstance(value, BaseException):
                job[-1].set_exception(value)
            else:
                job[-1].set_result(value)


def run_texts(state: dict, jobs: list[tuple[list[str], Future]]) -> list[np.ndarray]:
    """Every waiting query of every request in one call; each request gets its rows back."""
    batch = [t for request, _ in jobs for t in request]
    with tracer.start_as_current_span("embed.queries", attributes={"texts": len(batch)}):
        vectors = queries.embed(state["engine"].queries, batch)
    TEXTS.add(len(batch))
    ends = np.cumsum([len(request) for request, _ in jobs])
    return np.split(vectors, ends[:-1])


def run_pictures(state: dict, grid: Grid, jobs: list[tuple[float, torch.Tensor, Future]]) -> np.ndarray:
    """One engine call for up to max_batch pictures of one canvas, from any requests."""
    canvas = f"{grid[0]}x{grid[1]}"
    t0 = time.perf_counter()
    with tracer.start_as_current_span("embed.batch", attributes={"canvas": canvas, "pictures": len(jobs)}):
        vectors = embed(state["engine"], grid, [patches[None] for _, patches, _ in jobs]).cpu().numpy()
    BATCH_MS.record((time.perf_counter() - t0) * 1e3, {"canvas": canvas})
    PICTURES.add(len(jobs))
    return vectors


def submit_picture(state: dict, kind: str, source: str | bytes) -> Future:
    """On a CPU thread: check and prepare one picture and queue it at once; returns the future of its vector."""
    if kind == "path":
        # [Agent] The file opened is the resolved path that was checked, so a symlink cannot point the read elsewhere.
        path = Path(os.fsdecode(source)).resolve()
        if not state["settings"].allows(path):
            raise PermissionError("path outside EMBED_PATH_ROOTS")
        source = str(path)
    grid, patches = process(source)
    # [Agent] Pinned here, on the CPU thread, so the GPU thread's copy into the graph slot is a DMA, not a staged copy.
    patches = patches.pin_memory()
    future = Future()
    with state["ready"]:
        state["pictures"][grid].append((time.perf_counter(), patches, future))
        state["ready"].notify()
    return future


def submit_texts(state: dict, texts: list[str]) -> Future:
    future = Future()
    with state["ready"]:
        state["texts"].append((texts, future))
        state["ready"].notify()
    return future


async def embed_picture(state: dict, kind: str, source: str | bytes) -> np.ndarray:
    future = await asyncio.get_running_loop().run_in_executor(state["prep"], submit_picture, state, kind, source)
    return await asyncio.wrap_future(future)


async def embed_inputs(
    state: dict, items: list[Item], query: bool = False
) -> tuple[list[tuple[str, bytes]], list[tuple[str, str]]]:
    """(id, kind, source) inputs -> ([(id, float16 vector bytes)], [(id, error)]), both in input order. A picture that
    cannot be read, is over max_upload_mb or lies outside EMBED_PATH_ROOTS fails alone. Raises Busy (nothing queued)
    when the service is full."""
    settings = state["settings"]
    # [Agent] Results are kept by input position, never by id: two inputs may share an id (two uploads named photo.jpg).
    results: list[object] = [None] * len(items)
    pictures, texts = [], []
    for k, (_, kind, source) in enumerate(items):
        if kind == "text":
            texts.append(k)
        elif kind == "data" and len(source) > settings.max_upload_mb << 20:
            results[k] = ValueError(f"over {settings.max_upload_mb} MB")
        elif kind in ("path", "data"):
            pictures.append(k)
        else:
            results[k] = ValueError("an input needs data, path or text")
    if pictures and not query and free_mb() < settings.min_free_mb:
        raise Busy("not enough free card memory")
    # [Agent] Pictures count from admission until the request ends, so ones still decoding hold their place too.
    with state["ready"]:
        if state["inflight"] + len(pictures) > settings.max_queued:
            raise Busy(f"more than {settings.max_queued} pictures are in flight")
        state["inflight"] += len(pictures)
    try:
        jobs: list[Awaitable[Any]] = [embed_picture(state, *items[k][1:]) for k in pictures]
        if texts:
            jobs.append(asyncio.wrap_future(submit_texts(state, [items[k][2] for k in texts])))
        done = await asyncio.gather(*jobs, return_exceptions=True)
    finally:
        with state["ready"]:
            state["inflight"] -= len(pictures)
    for k, row in zip(pictures, done, strict=False):
        results[k] = row
    if texts:
        batch = done[-1]
        for n, k in enumerate(texts):
            results[k] = batch if isinstance(batch, BaseException) else batch[n]
    good = [k for k, row in enumerate(results) if isinstance(row, np.ndarray)]
    if good:
        rows = cut(np.stack([results[k] for k in good]), settings.dim)
        for k, row, finite in zip(good, rows.astype("<f2"), np.isfinite(rows).all(axis=1), strict=True):
            results[k] = row.tobytes() if finite else ValueError("the model gave a vector that is not finite")
    vectors = [(items[k][0], row) for k, row in enumerate(results) if isinstance(row, bytes)]
    failed = [
        (items[k][0], str(row) or type(row).__name__) for k, row in enumerate(results) if isinstance(row, BaseException)
    ]
    if failed:
        FAILURES.add(len(failed))
        log.info("embed.input.failed", extra={"failed": len(failed)})
    return vectors, failed
