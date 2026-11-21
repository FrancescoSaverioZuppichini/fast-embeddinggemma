"""One GPU thread serves every request, HTTP or gRPC: search queries first, then the canvas whose oldest picture has waited
longest, up to max_batch pictures from any requests in one engine call. Pictures are prepared on CPU threads before they queue.
The queue lives only in memory and only for in-flight work: durable queueing belongs to the caller (stash keeps it in its
database). Past max_queued waiting pictures a request gets Busy, which the caller answers as 503 / RESOURCE_EXHAUSTED."""

import asyncio
import logging
import threading
import time
from collections import deque
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
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
    """More pictures are waiting than EMBED_MAX_QUEUED, or the card is short of memory: try again later."""


def start(settings: Settings) -> dict:
    """Load the engine and its query graphs and start the GPU thread; every request works through the returned state."""
    engine = load(settings.precision, max_batch=settings.max_batch)
    engine["queries"] = queries.load(engine)
    state: dict[str, Any] = dict(
        engine=engine,
        settings=settings,
        prep=ThreadPoolExecutor(settings.threads),
        pictures={grid: deque() for grid in CANVASES},
        texts=deque(),
        ready=threading.Condition(),
        running=True,
    )
    meter.create_observable_gauge(
        "embed.queue.depth", callbacks=[lambda options: [Observation(queued(state))]], description="pictures waiting"
    )
    state["thread"] = threading.Thread(target=run, args=(state,), name="gpu", daemon=True)
    state["thread"].start()
    return state


def stop(state: dict) -> None:
    with state["ready"]:
        state["running"] = False
        state["ready"].notify()
    state["thread"].join()
    state["prep"].shutdown(cancel_futures=True)


def queued(state: dict) -> int:
    return sum(len(q) for q in state["pictures"].values())


def free_mb() -> int:
    return torch.cuda.mem_get_info()[0] >> 20


def health(state: dict) -> dict:
    settings, free = state["settings"], free_mb()
    return dict(
        model=settings.model,
        dims=settings.dim,
        freeVramMb=free,
        queue=dict(images=queued(state), texts=len(state["texts"])),
        vram="low" if free < settings.min_free_mb else "ok",
    )


def resolve(future: Future, value: object) -> None:
    """Hand a future its result or exception, unless its caller already gave up on it."""
    try:
        if isinstance(value, BaseException):
            future.set_exception(value)
        else:
            future.set_result(value)
    except InvalidStateError:
        pass


def run(state: dict) -> None:
    """The GPU thread: waits for work, takes all waiting queries or one picture batch, runs it outside the lock."""
    pictures, texts, max_batch = state["pictures"], state["texts"], state["settings"].max_batch
    while True:
        with state["ready"]:
            while state["running"] and not texts and not queued(state):
                state["ready"].wait()
            if not state["running"]:
                return
            if texts:
                grid, jobs = None, [texts.popleft() for _ in range(len(texts))]
            else:
                grid = min((g for g, q in pictures.items() if q), key=lambda g: pictures[g][0][0])
                jobs = [pictures[grid].popleft() for _ in range(min(len(pictures[grid]), max_batch))]
        if grid is None:
            run_texts(state, jobs)
        else:
            run_pictures(state, grid, jobs)


def run_texts(state: dict, jobs: list[tuple[list[str], Future]]) -> None:
    """Every waiting query of every request in one call; each request gets its rows back."""
    batch = [t for request, _ in jobs for t in request]
    with tracer.start_as_current_span("embed.queries", attributes={"texts": len(batch)}):
        try:
            vectors = queries.embed(state["engine"]["queries"], batch)
        except Exception as error:
            log.exception("embed.queries.failed", extra={"texts": len(batch)})
            for _, future in jobs:
                resolve(future, error)
            return
    TEXTS.add(len(batch))
    at = 0
    for request, future in jobs:
        resolve(future, vectors[at : at + len(request)])
        at += len(request)


def run_pictures(state: dict, grid: Grid, jobs: list[tuple[float, torch.Tensor, Future]]) -> None:
    """One engine call for up to max_batch pictures of one canvas, from any requests."""
    canvas = f"{grid[0]}x{grid[1]}"
    t0 = time.perf_counter()
    with tracer.start_as_current_span("embed.batch", attributes={"canvas": canvas, "pictures": len(jobs)}):
        try:
            vectors = embed(state["engine"], grid, [patches[None] for _, patches, _ in jobs]).cpu().numpy()
        except Exception as error:
            log.exception("embed.batch.failed", extra={"canvas": canvas, "pictures": len(jobs)})
            for *_, future in jobs:
                resolve(future, error)
            return
    BATCH_MS.record((time.perf_counter() - t0) * 1e3, {"canvas": canvas})
    PICTURES.add(len(jobs))
    for (_, _, future), vector in zip(jobs, vectors, strict=True):
        resolve(future, vector)


def prepare(source: str | bytes) -> tuple[Grid, torch.Tensor]:
    grid, patches = process(source)
    # [Agent] Pinned here, on the CPU threads, so the GPU thread's copy into the graph slot is a DMA, not a staged copy.
    return grid, patches.pin_memory()


def submit(state: dict, pictures: list[tuple[Grid, torch.Tensor]], texts: list[str]) -> list[Future]:
    """Queue one request's pictures (one future each) and its texts (one future for all); Busy if the queue is full."""
    futures = [Future() for _ in pictures]
    texts_future = Future()
    with state["ready"]:
        if queued(state) + len(pictures) > state["settings"].max_queued:
            raise Busy(f"more than {state['settings'].max_queued} pictures are waiting")
        now = time.perf_counter()
        for (grid, patches), future in zip(pictures, futures, strict=True):
            state["pictures"][grid].append((now, patches, future))
        if texts:
            state["texts"].append((texts, texts_future))
        state["ready"].notify()
    return futures + ([texts_future] if texts else [])


async def embed_inputs(
    state: dict, items: list[Item], query: bool = False
) -> tuple[list[tuple[str, bytes]], list[tuple[str, str]]]:
    """(id, kind, source) inputs -> ([(id, float16 vector bytes)], [(id, error)]), vectors in input order. A picture that
    cannot be read, or a path outside EMBED_PATH_ROOTS, fails alone. Raises Busy (nothing queued) when the service is full."""
    settings = state["settings"]
    failed: dict[str, str] = {}
    pictures, texts = [], []
    for i, kind, s in items:
        if kind == "text":
            texts.append((i, s))
        elif kind == "path" and not settings.allows(s):
            failed[i] = "path outside EMBED_PATH_ROOTS"
        elif kind in ("path", "data"):
            pictures.append((i, s))
        else:
            failed[i] = "an input needs data, path or text"
    if pictures and not query and free_mb() < settings.min_free_mb:
        raise Busy("not enough free card memory")
    loop = asyncio.get_running_loop()
    prepared = await asyncio.gather(
        *[loop.run_in_executor(state["prep"], prepare, s) for _, s in pictures], return_exceptions=True
    )
    ready = []
    for (i, _), result in zip(pictures, prepared, strict=True):
        if isinstance(result, BaseException):
            failed[i] = str(result) or type(result).__name__
        else:
            ready.append((i, result))
    futures = submit(state, [p for _, p in ready], [s for _, s in texts])
    results = await asyncio.gather(*[asyncio.wrap_future(f) for f in futures], return_exceptions=True)
    rows = {i: r for (i, _), r in zip(ready, results[: len(ready)], strict=True)}
    if texts:
        batch = results[-1]
        rows |= {i: (batch if isinstance(batch, BaseException) else batch[k]) for k, (i, _) in enumerate(texts)}
    vectors = []
    for i, _, _ in items:
        row = rows.get(i)
        if isinstance(row, BaseException):
            failed[i] = str(row) or type(row).__name__
        elif row is not None and np.isfinite(row).all():
            vectors.append((i, cut(row[None], settings.dim)[0].astype("<f2").tobytes()))
        elif row is not None:
            failed[i] = "the model gave a vector that is not finite"
    if failed:
        FAILURES.add(len(failed))
        log.info("embed.input.failed", extra={"failed": len(failed)})
    return vectors, list(failed.items())
