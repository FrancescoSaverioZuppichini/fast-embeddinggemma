"""HTTP service with the contract of the embedding sidecar it replaces: POST /embed {model, inputs: [{id, path} | {id, text}], query}
-> {model, dims, vectors: [{id, vec}], failed: [{id, error}]} and GET /health. Pictures are prepared on CPU threads, then
merged across requests into one engine call per canvas on ONE GPU thread; text queries run the stock Hugging Face text tower.
Settings are EMBED_* environment variables. Run: fast-embeddinggemma serve."""

import asyncio
import base64
import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import __version__, model, prep

ENV = os.environ.get
CFG = {
    "model": ENV("EMBED_MODEL", "google/embeddinggemma-2"),
    "dim": int(ENV("EMBED_DIM", "768")),
    "min_free_mb": int(ENV("EMBED_MIN_FREE_MB", "900")),
    "max_batch": int(ENV("EMBED_MAX_BATCH_IMAGES", "32")),
    "threads": int(ENV("EMBED_DECODE_THREADS", "8")),
    "precision": ENV("EMBED_PRECISION", "int8"),
}
QUERY_PROMPT = "task: search result | query: "
STATE = {}


class Input(BaseModel):
    id: str
    path: str | None = None
    text: str | None = None


class EmbedRequest(BaseModel):
    model: str
    inputs: list[Input]
    query: bool = False


def log(**fields):
    print(json.dumps({"t": round(time.time(), 3), **fields}), flush=True)


def free_mb():
    return torch.cuda.mem_get_info()[0] >> 20


def load_text():
    """Tokenizer + the query encoder: only the Hugging Face language model goes to the card, the vision tower never does.
    Eager on purpose: torch.compile made a warm query 4 ms instead of 25, but start-up 99 s instead of 22, and each new query length
    recompiled for up to 85 s."""
    from transformers import AutoModel, AutoTokenizer

    hf = AutoModel.from_pretrained(CFG["model"], dtype=torch.bfloat16, attn_implementation="sdpa", audio_config=None)
    lm = hf.language_model.to("cuda").eval()

    def encode(ids, mask):
        h = lm(inputs_embeds=lm.embed_tokens(ids), attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return F.normalize((h.float() * m).sum(1) / m.sum(1), dim=-1)

    return AutoTokenizer.from_pretrained(CFG["model"]), encode


def embed_texts(texts):
    """Search queries on the GPU thread -> one vector, or the error, per text."""
    try:
        with torch.inference_mode():
            tok = STATE["tok"](texts, padding=True, truncation=True, max_length=8192, return_tensors="pt")
            return list(STATE["text"](tok["input_ids"].cuda(), tok["attention_mask"].cuda()).float().cpu().numpy())
    except Exception as error:  # noqa: BLE001
        return [error] * len(texts)


def prep_picture(path):
    grid, patches = prep.read(path)
    # [Agent] Pinned here, on the prep threads, so the GPU thread's copy into the graph slot is a DMA, not a staged copy.
    return grid, patches.pin_memory()


def answer(fut, value):
    """From the GPU thread: hand a request its vector (or error), unless the request was cancelled meanwhile."""

    def done():
        if not fut.done():
            fut.set_result(value)

    fut.get_loop().call_soon_threadsafe(done)


def take_batch():
    """Under the lock: the canvas whose oldest picture waited longest, up to max_batch of its pictures, from any requests."""
    queues = STATE["queues"]
    grid = min((g for g, q in queues.items() if q), key=lambda g: queues[g][0][0])
    return grid, [queues[grid].popleft() for _ in range(min(len(queues[grid]), CFG["max_batch"]))]


def drain():
    """The GPU thread: engine calls until nothing waits. No waiting window: whatever is ready runs."""
    while True:
        with STATE["lock"]:
            if not any(STATE["queues"].values()):
                return
            grid, batch = take_batch()
        t0 = time.perf_counter()
        try:
            vecs = list(model.embed(STATE["engine"], grid, [p[None] for _, p, _ in batch]).cpu().numpy())
        except Exception as error:  # noqa: BLE001
            vecs = [error] * len(batch)
        for (_, _, fut), vec in zip(batch, vecs, strict=True):
            answer(fut, vec)
        log(event="batch", canvas=list(grid), n=len(batch), ms=round((time.perf_counter() - t0) * 1e3, 2))


async def embed_pictures(items):
    """Prepare every picture of the request first, then queue them together so a batch can hold the whole request."""
    loop = asyncio.get_running_loop()
    pictures = await asyncio.gather(
        *[loop.run_in_executor(STATE["prep"], prep_picture, i.path) for i in items], return_exceptions=True
    )
    futs = [loop.create_future() for _ in items]
    with STATE["lock"]:
        for pic, fut in zip(pictures, futs, strict=True):
            if isinstance(pic, BaseException):
                fut.set_result(pic)
            else:
                STATE["queues"][pic[0]].append((time.perf_counter(), pic[1], fut))
    STATE["gpu"].submit(drain)
    return dict(zip([i.id for i in items], await asyncio.gather(*futs), strict=True))


async def embed_queries(items):
    vecs = await asyncio.wrap_future(STATE["gpu"].submit(embed_texts, [QUERY_PROMPT + i.text for i in items]))
    return dict(zip([i.id for i in items], vecs, strict=True))


def encode_vectors(results):
    """{id: vector or error} -> (vectors, failed): cut to EMBED_DIM, unit length again, float16, base64, in one numpy pass."""
    good = [k for k, r in results.items() if not isinstance(r, BaseException)]
    failed = [
        {"id": k, "error": str(r) or type(r).__name__} for k, r in results.items() if isinstance(r, BaseException)
    ]
    if not good:
        return [], failed
    cut = np.stack([results[k] for k in good])[:, : CFG["dim"]]
    norm = np.linalg.norm(cut, axis=1)
    ok = np.isfinite(cut).all(1) & np.isfinite(norm) & (norm > 0)
    half = (cut / np.where(ok, norm, 1)[:, None]).astype(np.float16)
    failed += [
        {"id": k, "error": "The model gave a vector that is not finite"} for k, o in zip(good, ok, strict=True) if not o
    ]
    vectors = [
        {"id": k, "vec": base64.b64encode(v.tobytes()).decode()} for k, v, o in zip(good, half, ok, strict=True) if o
    ]
    return vectors, failed


@asynccontextmanager
async def lifespan(_):
    if ENV("EMBED_IMAGE_TOKENS", "140") != "140":
        raise ValueError("the engine's canvases are built for 140 image tokens")
    started = time.time()
    tok, text = load_text()
    STATE.update(
        engine=model.load("cuda", CFG["precision"]),
        tok=tok,
        text=text,
        lock=threading.Lock(),
        queues={g: deque() for g in model.canvases()},
        prep=ThreadPoolExecutor(CFG["threads"]),
        gpu=ThreadPoolExecutor(1),
    )
    STATE["gpu"].submit(embed_texts, [QUERY_PROMPT + "warm up"] * 2).result()
    log(event="ready", load_s=round(time.time() - started, 2), cfg=CFG, version=__version__)
    yield
    STATE["prep"].shutdown(cancel_futures=True)
    STATE["gpu"].shutdown(cancel_futures=True)


app = FastAPI(lifespan=lifespan)


@app.post("/embed")
async def embed(request: EmbedRequest):
    if request.model != CFG["model"]:
        raise HTTPException(409, f"This service runs {CFG['model']}, not {request.model}")
    if any((i.path is None) == (i.text is None) for i in request.inputs):
        raise HTTPException(422, "an input is {id, path} or {id, text}")
    if not request.query and free_mb() < CFG["min_free_mb"]:
        raise HTTPException(503, "Not enough free video memory")
    pictures = [i for i in request.inputs if i.path is not None]
    texts = [i for i in request.inputs if i.text is not None]
    step = CFG["max_batch"]
    jobs = [embed_pictures(pictures[k : k + step]) for k in range(0, len(pictures), step)]
    parts = await asyncio.gather(*jobs, *([embed_queries(texts)] if texts else []))
    results = {k: v for part in parts for k, v in part.items()}
    vectors, failed = encode_vectors({key: results[key] for key in dict.fromkeys(i.id for i in request.inputs)})
    return {"model": CFG["model"], "dims": CFG["dim"], "vectors": vectors, "failed": failed}


@app.get("/health")
async def health():
    free = free_mb()
    with STATE["lock"]:
        queued = sum(len(q) for q in STATE["queues"].values())
    return {
        "model": CFG["model"],
        "dims": CFG["dim"],
        "freeVramMb": free,
        "queue": {"images": queued, "texts": 0},
        "vram": "low" if free < CFG["min_free_mb"] else "ok",
    }
