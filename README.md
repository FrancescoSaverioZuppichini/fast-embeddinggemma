# fast-embeddinggemma

Fast image embeddings with Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) on an NVIDIA card.
It's a small PyTorch + Triton engine for the image path at 140 soft tokens. Every picture is snapped to one of 7 fixed
canvases, so each (canvas, batch size) pair runs as one CUDA graph with no padding. The vision tower's matrix products run in INT8 by default.
You get 768-number unit vectors that search against the stock model's text vectors.

## Numbers

RTX 3090 at 300 W, 140 soft tokens, library thumbnails. "Stock" is the Hugging Face model under `torch.compile`
(max-autotune) with SageAttention. The last column names the benchmark run each number comes from.

| | stock | INT8 (default) | bf16 | run |
|---|---|---|---|---|
| model alone, batch 32, img/s | 131.8 | 215.2 (1.62x) | 156.2 (1.19x) | B3 |
| HTTP service, 32 pictures per request, 2 in flight, img/s | 92.5 | 213.3 (2.31x) | 153.2 (1.66x) | I2 |
| HTTP service, 1 picture per request, img/s | 36.1 | 134.4 (3.7x) | 104.6 | I2 |
| search R@10 against the fp32 reference | -0.44 pt | -1.54 pt | -0.69 pt | B3 |
| cold start | 509.5 s (compile) | 15-18 s | | B4 |

The service rows were measured with the v2 service and its native picture preparation. This package's server prepares
pictures with Pillow on CPU threads instead, which is fast enough that the card stays the limit (see `fast-embeddinggemma bench`).

**What INT8 costs:** recall@10 drops by 1.54 points (bf16: 0.69). That's measured on a frozen set of 2,200 pictures and 50
judged text queries, against fp32 vectors, so it moves by about a point either way. If search quality matters more than
speed, use bf16: `load(precision="bf16")`, or `EMBED_PRECISION=bf16` for the server. Vectors from different precisions are
close but not identical, so don't mix them in one index.

## Install

Linux x86_64 with an NVIDIA card. The numbers above are for torch 2.14.0 built for CUDA 13; the dependencies pin the
versions the vectors were measured on.

```bash
uv add git+https://github.com/FrancescoSaverioZuppichini/fast-embeddinggemma            # the engine
uv add "fast-embeddinggemma[server] @ git+https://github.com/FrancescoSaverioZuppichini/fast-embeddinggemma"  # + HTTP service
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

The engine reads the weights from the local Hugging Face cache and never downloads them. Accept the Gemma terms on the
model page first, then run `hf download google/embeddinggemma-2`.

## Use

```python
import fast_embeddinggemma as feg

engine = feg.load()  # INT8; feg.load(precision="bf16") for vectors closer to the reference
vectors = feg.embed_images(engine, ["a.jpg", "b.webp", open("c.png", "rb").read()])
print(vectors.shape)  # (3, 768) float32, unit length: cosine similarity is a dot product

# one level down: prepare a picture, then embed many of one canvas in one call (vectors stay on the card)
grid, patches = feg.read("a.jpg")  # (rows, cols), uint8 (9 * rows * cols, 768)
on_card = feg.embed(engine, grid, patches[None])  # (1, 768)
```

- `load(device="cuda", precision="int8", checkpoint=None)` loads the weights and builds 84 CUDA graphs (7 canvases x 12 batch sizes). `checkpoint` is an HF-layout safetensors file and defaults to the cache.
- `embed(engine, grid, pieces)` takes uint8 patches of one canvas, as one tensor or a list of tensors. They can live in pinned memory or on the card, and they're copied straight into the graph's input. Any batch size works: each call takes at most two graph replays up to 32 pictures. Call it from one thread only.
- `embed_images(engine, paths_or_bytes)` prepares the pictures, groups them by canvas and returns vectors in input order.
- `read(path_or_bytes)` prepares a picture exactly the way the stock sidecar's decode feeds the model: Pillow draft + EXIF turn + Lanczos thumbnail to 1,135 px, then torchvision's uint8 bicubic antialias squash onto the nearest canvas.
- `canvases()` returns the 7 grids, as (rows, cols).

## CLI

```bash
fast-embeddinggemma bench ./pictures              # model-only img/s at 1, 8 and 32 pictures per call
fast-embeddinggemma serve --port 8100             # the HTTP service (needs the [server] extra)
fast-embeddinggemma convert-google bundle.litertlm out.safetensors   # needs the [google] extra
```

`convert-google` turns Google's QAT LiteRT-LM bundle into an HF-layout checkpoint that `load(checkpoint=...)` accepts.
It's a test tool: in our measurement Google's QAT model searched worse (R@10 -3.43 pt) than this package's own INT8.

## HTTP service

- `POST /embed {model, inputs: [{id, path} | {id, text}], query}` returns `{model, dims, vectors: [{id, vec}], failed: [{id, error}]}`. `vec` is a base64 little-endian float16 unit vector.
- Error codes: `409` if the request names another model; `503` if free video memory is under the floor (not checked for `query` requests); `422` for a malformed input.
- `GET /health` returns `{model, dims, freeVramMb, queue: {images, texts}, vram: "ok" | "low"}`.

How it handles work:
- Pictures are prepared on CPU threads, then batched across requests on one GPU thread. Each batch takes the canvas whose oldest picture has waited longest, with no waiting window.
- Text is always a search query. It gets the query prompt and runs the stock Hugging Face text tower; only the language model goes to the card. Moving queries onto the engine's own text tower would drop `transformers` and about 0.9 GB of card memory. That's a later milestone.
- The engine's memory is fixed at load, so there is no per-process memory cap.
- It opens any path it is sent, so keep it on 127.0.0.1.

| setting | default | |
|---|---|---|
| `EMBED_MODEL` | `google/embeddinggemma-2` | the model a request must name |
| `EMBED_DIM` | `768` | vector length (768, 512, 256 or 128: a cut, then unit length again) |
| `EMBED_PRECISION` | `int8` | `int8` or `bf16` |
| `EMBED_MIN_FREE_MB` | `900` | free video memory under which picture requests get 503 |
| `EMBED_MAX_BATCH_IMAGES` | `32` | pictures per engine call |
| `EMBED_DECODE_THREADS` | `8` | CPU threads that prepare pictures |
| `EMBED_COMPILE` | `max-autotune-no-cudagraphs` | `torch.compile` mode of the query tower, or `none` |

## Limits

- 140 soft tokens only. The 7 canvases cover about 96% of pictures with the grid the stock processor would have picked.
- The INT8 tile sizes were tuned on an RTX 3090 (sm_86). Other cards run, but may want other tiles.

## Develop

```bash
uv sync --all-extras
uv run ruff check && uv run ruff format --check
uv run pytest -m "not gpu"                      # prep against the reference + every kernel compiled for sm_86
FEG_PICTURES=/your/pictures uv run pytest -m gpu  # the engine on the card
```

`tests/prep_reference.json` holds the canvas and the patch hash that the previous native preparation produced for seeded
synthetic pictures (these run everywhere) and for library pictures. Library pictures are listed only by the hash of their
file, and those tests run when `FEG_PICTURES` points at a folder that holds them.

## License

The code is Apache-2.0. The EmbeddingGemma weights are Google's and follow the [Gemma terms of use](https://ai.google.dev/gemma/terms).
