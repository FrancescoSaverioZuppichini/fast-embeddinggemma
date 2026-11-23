# fast-embeddinggemma

Fast embeddings with Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) on an NVIDIA card. Pictures
and search queries go into one 768-number space of unit vectors, so cosine similarity is a dot product. It's a small
PyTorch + Triton engine with a FastAPI + gRPC service on top. INT8 by default, 140 image tokens. Search quality is
measured on our public benchmark, [aspect-retrieval-5k](https://huggingface.co/datasets/Francesco/aspect-retrieval-5k).

## Numbers

Every number below comes from this one machine:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GB, power limit 300 W (stock 350 W) |
| CPU | AMD Ryzen 9 5950X, 16 cores / 32 threads |
| RAM | 64 GB DDR4-3200 |
| OS | Ubuntu 22.04.5, Linux 6.8 |
| Software | NVIDIA driver 595.99, CUDA 13.2, torch 2.14.0 (cu130 wheels), Triton 3.8.0, Python 3.12 |

"HF stack" is the Hugging Face model under `torch.compile` (max-autotune) with
SageAttention: the sidecar this package replaced.

**The model alone** (pictures a second, files already in memory):

| tokens | pictures per call | HF stack | ours, bf16 | ours, INT8 (default) |
|---|---|---|---|---|
| 140 | 1 | 67.6 | 102.1 | **130.0** (1.92x) |
| 140 | 8 | 123.1 | 146.1 | **200.7** (1.63x) |
| 140 | 32 | 131.8 | 156.2 | **215.2** (1.62x) |
| 280 | 1 | 47.9 | 63.1 | **82.0** (1.72x) |
| 280 | 8 | 61.3 | 75.0 | **99.7** (1.64x) |
| 280 | 32 | 64.1 | 76.8 | **101.5** (1.59x) |

The package serves 140 tokens; the 280 rows come from the same engine with a 280-token canvas.

**The service** (files in, vectors out, 256 library thumbnails, 2 requests in flight):

| pictures per request | old HF sidecar | ours, HTTP | ours, gRPC (bytes) |
|---|---|---|---|
| 32 | 92.5 | **203-213** | 199 |
| 8 | 80.5 | **188-199** | |
| 1 | 36.1 | **142** | |

**Search queries** (one query per call):

| query length | HF text model | ours |
|---|---|---|
| up to 512 tokens | 26.6 ms | **2.0 ms** (13x) |
| 513 to 1,024 | 26.2 ms | 12.9 ms |
| 1,025 to 2,048 | 40.1 ms | 27.1 ms |
| 2,049 to 8,192 | 70.1 ms | 48.8 ms |

**Start-up, memory**

| | HF stack | ours |
|---|---|---|
| cold start | 509 s (compile) | about 10 s (engine) / 15-21 s (service, 21 with a cold kernel cache) |
| card memory, service | | 3.1 GB, the whole process as nvidia-smi counts it (4.5 GB without expandable segments) |

**Search quality** on [aspect-retrieval-5k](https://huggingface.co/datasets/Francesco/aspect-retrieval-5k), our public
benchmark: 5,040 Open Images photos, 720 on each of the 7 canvases, each one looked for among 25,040 photos by a typed
query, a sentence, a person's description, a re-posted copy and a line drawing (`fast-embeddinggemma eval`):

| | short R@1 | sentence R@1 | human R@10 | near-dup R@1 | sketch R@10 |
|---|---|---|---|---|---|
| INT8 (default) | 0.758 | 0.934 | 0.451 | 0.902 | 0.222 |
| bf16 | 0.757 | 0.934 | 0.450 | 0.902 | 0.221 |

The 95% intervals are about ±0.01, so INT8 costs nothing measurable here. Vectors from different precisions are close
but not identical, so don't mix them in one index. Per shape (INT8), the widest photos are the hardest to find by text:

| | 9:16 `15x8` | 2:3 `14x9` | 3:4 `13x10` | 1:1 `11x11` | 4:3 `10x13` | 3:2 `9x14` | 16:9 `8x15` |
|---|---|---|---|---|---|---|---|
| short R@1 | 0.824 | 0.796 | 0.793 | 0.775 | 0.733 | 0.726 | 0.661 |
| sentence R@1 | 0.960 | 0.960 | 0.954 | 0.947 | 0.914 | 0.925 | 0.876 |
| human R@10 | 0.525 | 0.482 | 0.503 | 0.447 | 0.403 | 0.449 | 0.346 |
| near-dup R@1 | 0.919 | 0.929 | 0.906 | 0.878 | 0.881 | 0.893 | 0.907 |
| sketch R@10 | 0.265 | 0.199 | 0.233 | 0.236 | 0.182 | 0.188 | 0.250 |

The intervals per shape are on the [dataset card](https://huggingface.co/datasets/Francesco/aspect-retrieval-5k).

## How it's fast

1. **Its own model code.** Plain PyTorch modules (`vision.py`, `text.py`, `model.py`) that read like Hugging Face's
   modeling files and call fused Triton kernels, with the reference's maths. There's no `transformers` at runtime.
2. **Seven fixed canvases.** At 140 tokens every picture goes onto one of 7 grids, chosen by aspect ratio. Shapes are
   static, so nothing is padded and nothing is masked.
3. **CUDA graphs.** The whole forward pass is recorded once per (canvas, batch size), and a batch is one replay. There's
   no `torch.compile`, so start-up takes seconds.
4. **Fused kernels.** These each do in one memory trip what took three to five:
   - q/k RMS norm plus 2-D rotary positions;
   - residual add plus post-norm, plus the next pre-norm and the layer scale;
   - GELU(gate) × up.
5. **INT8 matrix products** in the vision tower: INT8 weights per channel, and activations in 64-value groups.
   - The QKV product runs norm and RoPE in its epilogue.
   - The gate|up product runs GELU and INT8 packing in its epilogue.
   - The next pre-norm is folded into the quantization factor.
6. **Queries through the same text tower.** There's one graph per length bucket, and the 256 MiB embedding table stays
   in host memory.
7. **The service keeps the card busy.**
   - Each picture is prepared on a CPU thread and joins the GPU queue the moment it's ready.
   - One GPU thread batches pictures from all requests by canvas, up to 32 per call, and runs queries first.
   - A picture travels as uint8 (4x smaller than float32) in pinned memory: one copy into the graph's input.

**What we tried that didn't help:**
- a Rust engine;
- SageAttention;
- INT4;
- merging image tokens;
- a small trained head instead of the text tower;
- batches over 32;
- two batches in flight on two CUDA streams.

At 300 W this card is power-bound, so only doing less work per picture moves the number.

## Install

Linux x86_64 with an NVIDIA card (Ampere or newer), torch 2.14 for CUDA 13.

```bash
uv add git+https://github.com/FrancescoSaverioZuppichini/fast-embeddinggemma                       # the engine
uv add "fast-embeddinggemma[server] @ git+https://github.com/FrancescoSaverioZuppichini/fast-embeddinggemma"  # + the service
```

torch comes from the PyTorch index. Put this in your own `pyproject.toml` (uv only reads index settings from the root project):

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

The weights download on first use from [Francesco/fast-embeddinggemma](https://huggingface.co/Francesco/fast-embeddinggemma):
Google's checkpoint converted once into this package's names (q|k|v and gate|up fused, the token table and tokenizer
included, the audio tower left out), 878 MB instead of Google's full file, with bit-identical vectors. The repo is
private: log in with an account that can read it (`hf auth login`, or `HF_TOKEN`). With `HF_HUB_OFFLINE=1` the package
never touches the network.

To build the weights yourself from [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (gated: accept
the Gemma terms there first), run `fast-embeddinggemma convert ./weights` and pass
`load(weights="./weights/model.safetensors")`. The weights stay under the
[Gemma terms of use](https://ai.google.dev/gemma/terms).

## Use it from Python

```python
import fast_embeddinggemma as feg

model = feg.load()  # INT8, about 10 s; feg.load("bf16") for vectors closer to the reference
images = feg.embed_images(model, ["cat.jpg", png_bytes, pil_image, rgb_array])  # (4, 768) float32, unit length
query = feg.embed_texts(model, ["a cat on a sofa"])  # (1, 768), same space
scores = query @ images.T  # cosine similarity
```

- **`load(precision="int8", weights=None, max_batch=32)`** loads the weights and records one CUDA graph per canvas and
  batch size. `weights` is a file made by `fast-embeddinggemma convert` (default: the published one). `max_batch` records
  bigger graphs: 64 and 128 gain under 1% here and cost about 1.4 GB per doubling. The model itself is
  `engine.model`, a plain `torch.nn.Module` (`EmbeddingGemma2`) you can call eagerly: `model.embed_images(patches, canvas)`.
- **`embed_images(model, images, dim=768, threads=8)`** takes paths, bytes, PIL images or uint8 RGB arrays, any number of
  them. It prepares 256 at a time on CPU threads while the card works, and returns vectors in input order. A picture
  that can't be read raises `ValueError` with its position. `dim` of 512, 256 or 128 cuts the vector (Matryoshka) and
  makes it unit length again.
- **`embed_texts(model, texts, dim=768)`** embeds search queries, with the query prompt added.
- **`process(image)`** prepares one picture: `(canvas, uint8 patches)`. **`embed(model, canvas, patches)`** embeds
  pictures of one canvas in one call; the vectors stay on the card.

## Command line

```bash
fast-embeddinggemma embed ./photos --out vectors.npy   # every picture under a folder; the paths go to vectors.txt
fast-embeddinggemma bench --sizes 1 8 32              # model-only pictures a second, on the dataset or ./photos
fast-embeddinggemma eval --precision int8             # search quality on aspect-retrieval-5k
fast-embeddinggemma serve                              # the service (needs the [server] extra)
fast-embeddinggemma convert ./weights                  # Google's checkpoint -> this package's weights + tokenizer
```

`eval`, and `bench` without a folder, download [aspect-retrieval-5k](https://huggingface.co/datasets/Francesco/aspect-retrieval-5k)
at its pinned version (`--revision v1.0`) and need the `eval` extra (`fast-embeddinggemma[eval]`). `--data DIR` reads a
copy saved with `save_to_disk` instead.

## The service

`fast-embeddinggemma serve` runs HTTP and gRPC in one process. There is one GPU thread for both. The OpenAPI page is at
**`/docs`**, and the schema at `/openapi.json`.

| HTTP | |
|---|---|
| `POST /embed` | JSON `{model, inputs: [{id, path} or {id, text}], query}`: pictures by path on the server's disk, and search queries |
| `POST /embed/files` | multipart: `files` (the pictures' bytes) and optional `ids`, for clients on another machine |
| `GET /health` | `{model, dims, freeVramMb, queue: {images, texts}, vram: "ok" or "low"}`; `model` is the tag, e.g. `google/embeddinggemma-2@int8` |

Requests name the tag `/health` reports (`checkpoint@precision`): a new precision is a new vector space, so a client
that stored vectors of another tag gets 409 instead of mixing them. Both embed calls answer
`{model, dims, vectors: [{id, vec}], failed: [{id, error}]}`. `vec` is base64 of `dims`
little-endian float16 values, unit length. A picture that can't be read fails alone, with its reason. The error codes:
- `409`: the request names another model;
- `413`: too many uploads, or one over `EMBED_MAX_UPLOAD_MB`;
- `422`: a malformed body;
- `503`: busy, so retry later.

**gRPC**, on `EMBED_GRPC_PORT` ([`embed.proto`](src/fast_embeddinggemma/server/embed.proto)): `Embed` takes inputs with
`data` (bytes), `path` or `text`; vectors come back as raw float16 bytes. `Health` mirrors `/health`. A `data` input
over `EMBED_MAX_UPLOAD_MB` fails alone, and a message is capped at one full batch of such pictures (at most 1 GiB).

**How it handles work:**
- One GPU thread runs waiting search queries first. Then it takes the canvas whose oldest picture has waited longest,
  up to `EMBED_MAX_BATCH` pictures from any requests, in one engine call. There is no waiting window.
- Each picture joins the queue as soon as its CPU thread has decoded it, so the card starts on a big request while
  the rest still decode.
- More than `EMBED_MAX_QUEUED` pictures in flight (decoding or waiting), or free card memory under
  `EMBED_MIN_FREE_MB` (search queries are exempt), answers 503 or `RESOURCE_EXHAUSTED`.
- Shutdown finishes the pictures already admitted; a request whose caller gave up is dropped before the card runs it.
- `serve` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless you set it: 3.1 GB of card memory instead
  of 4.5, at the same speed.
- The queue only holds work in flight. Durable queueing belongs to the caller: stash keeps it in its database.
- Paths are read only under `EMBED_PATH_ROOTS`, checked on the resolved path that is then opened. Unset, no path is
  read: send bytes.

| setting | default | |
|---|---|---|
| `EMBED_PRECISION` | `int8` | `int8` or `bf16` |
| `EMBED_DIM` | `768` | 768, 512, 256 or 128 |
| `EMBED_HOST` / `EMBED_PORT` / `EMBED_GRPC_PORT` | `127.0.0.1` / `8100` / `8101` | a port of 0 turns gRPC off |
| `EMBED_PATH_ROOTS` | empty (no paths) | `:`-separated folders whose files may be sent by path |
| `EMBED_MAX_BATCH` | `32` | most pictures per engine call (32, 48, 64, 96, 128) |
| `EMBED_MAX_QUEUED` | `1024` | pictures in flight before 503 |
| `EMBED_MAX_UPLOAD_MB` | `32` | per picture sent as bytes (HTTP and gRPC) |
| `EMBED_MIN_FREE_MB` | `900` | free card memory under which picture requests get 503 |
| `EMBED_THREADS` | `8` | CPU threads preparing pictures |

### Docker

```bash
docker build -t fast-embeddinggemma .
docker run --gpus all -p 127.0.0.1:8100:8100 -p 127.0.0.1:8101:8101 \
  -v hf-cache:/cache -e HF_TOKEN=$HF_TOKEN \
  -v /photos:/photos:ro -e EMBED_PATH_ROOTS=/photos \
  fast-embeddinggemma
```

The container listens on all interfaces, so publish its ports on `127.0.0.1` unless you mean to expose it. Weights and
compiled kernels go to the `/cache` volume. Mount picture folders read-only and list them in `EMBED_PATH_ROOTS` to send
paths; otherwise send bytes.

### Events (OpenTelemetry)

The service emits spans (`embed.batch`, `embed.queries`), metrics (`embed.pictures`, `embed.texts`, `embed.failures`,
`embed.batch.duration`, `embed.queue.depth`) and log events (`server.start.completed`, `embed.batch.failed`, ...) through
the OpenTelemetry API only (the `server` extra). With nothing configured the spans and metrics cost nothing, and
`serve` prints the log events to the console. To export all of them, run it under the standard agent:

```bash
uv add opentelemetry-distro opentelemetry-exporter-otlp && uv run opentelemetry-bootstrap -a install
OTEL_SERVICE_NAME=fast-embeddinggemma OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317 \
OTEL_TRACES_EXPORTER=otlp OTEL_METRICS_EXPORTER=otlp OTEL_LOGS_EXPORTER=otlp \
OTEL_PYTHON_EXCLUDED_URLS=health uv run opentelemetry-instrument fast-embeddinggemma serve
```

## Limits

- 140 image tokens only. The 7 canvases cover about 96% of pictures with the grid the Hugging Face processor would pick.
- The INT8 tile sizes were tuned on an RTX 3090 (sm_86). Other cards run, but may want other tiles.
- One process uses one card. Run one per card.

## Code

Read the model in this order, each file top to bottom: `vision.py`, `text.py`, `model.py`.

```
src/fast_embeddinggemma/
  __init__.py        load, embed_images, embed_texts, process, embed, CANVASES
  preprocessing.py   any picture -> canvas + uint8 patches (CPU)
  cli.py, bench.py   fast-embeddinggemma embed | bench | eval | serve | convert
  quality.py         eval: R@1/5/10 and MRR on aspect-retrieval-5k, per query kind and canvas
  modeling/
    vision.py        Config, VisionLayer, Int8VisionLayer, VisionTower: patches -> 16 layers -> 3x3 average -> image tokens
    text.py          Config, TextLayer, TextTower: tokens -> 24 layers -> 768 numbers per token
    model.py         EmbeddingGemma2 = vision + text: embed_images, embed_queries; load_model
    canvas.py        the 7 canvases, which one a picture goes on, their position and RoPE tables
    engine.py        load (model + CUDA graphs per canvas and batch size) and embed
    queries.py       search queries through the text tower, one graph per length bucket
    convert.py       Google's checkpoint -> this package's weights (the only file that knows its names)
    kernels/         the Triton kernels: norm.py (fused add + norm), attention.py (q/k norm + RoPE), mlp.py (GELU x up), int8.py
  server/            the service
    config.py        settings (EMBED_*)
    schemas.py       the HTTP contract (the OpenAPI page)
    batcher.py       the one GPU thread
    main.py          FastAPI: /embed, /embed/files, /health
    rpc.py           gRPC (embed.proto)
```

## Develop

```bash
uv sync --all-extras
uv run ruff check && uv run ruff format --check
uv run pytest -m "not gpu"                        # preparation, settings, schemas, every kernel compiled for sm_86
FEG_PICTURES=/your/pictures uv run pytest -m gpu  # the engine, the queries and the service on the card
uv run --group dataset scripts/build_dataset.py --work DIR  # rebuild aspect-retrieval-5k (OPENROUTER_API_KEY, about $1)
uv run python -m grpc_tools.protoc -I src --python_out=src --pyi_out=src --grpc_python_out=src \
  src/fast_embeddinggemma/server/embed.proto      # after editing the proto
```

`tests/prep_reference.json` holds the canvas and patch hash of every reference picture. Library pictures appear only as
the hash of their file, and those tests run when `FEG_PICTURES` points at a folder that holds them.

## License

The code is Apache-2.0. The EmbeddingGemma weights are Google's and follow the [Gemma terms of use](https://ai.google.dev/gemma/terms).
