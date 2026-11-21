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
pictures with Pillow on CPU threads instead (2,097 img/s on 8 threads), and on the same 256 thumbnails it does 208.7 img/s at
32 pictures per request with 2 in flight (-2.2%), and 133.7 at 1 picture per request. It starts in 22 s.

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

The weights download into the Hugging Face cache on first use. They're gated: accept the Gemma terms on the
[model page](https://huggingface.co/google/embeddinggemma-2) and log in once with `hf auth login`. With
`HF_HUB_OFFLINE=1` it never touches the network.

## Use

```python
import fast_embeddinggemma as feg

model = feg.load()  # INT8, ~10 s; feg.load(precision="bf16") for vectors closer to the reference
images = feg.embed_images(model, ["cat.jpg", png_bytes, pil_image, rgb_array])  # (4, 768) float32, unit length
query = feg.embed_texts(model, ["a cat on a sofa"])  # (1, 768), same space, ~2 ms
scores = query @ images.T  # cosine similarity

# one level down: prepare one picture, then embed many of one canvas in one call (vectors stay on the card)
grid, patches = feg.process("cat.jpg")  # (rows, cols), uint8 (9 * rows * cols, 768)
on_card = feg.embed(model, grid, patches[None])  # (1, 768)
```

- `load(device="cuda", precision="int8", checkpoint=None)` loads the weights and builds 84 CUDA graphs (7 canvases x 12 batch sizes). `checkpoint` is an HF-layout safetensors file; by default the hub checkpoint.
- `embed_images(model, images, dim=768, threads=8)` takes paths, bytes, PIL images or uint8 RGB arrays, any number of them. They're prepared 256 at a time on CPU threads while the card works, grouped by canvas, and the vectors come back in input order. A picture that can't be read raises an error naming its index. `dim` of 512, 256 or 128 cuts the vector and makes it unit length again.
- `embed_texts(model, texts, dim=768)` embeds search queries through the engine's own text tower, with the query prompt added: one recorded CUDA graph per length bucket (16 to 512 tokens), about 2 ms per query, and the same vectors as the Hugging Face text model (cosine 0.99996 on 1,000 public texts). The graphs are built on the first call (about 2 s). Queries over 512 tokens are cut at 512.
- `process(image)` prepares one picture exactly the way the stock sidecar's decode feeds the model: Pillow (with the JPEG draft scale-down for files and bytes) + EXIF turn + Lanczos thumbnail to 1,135 px, then torchvision's uint8 bicubic antialias squash onto the nearest canvas. A path, bytes, a PIL image and an array of the same picture give the same patches.
- `embed(model, grid, pieces)` takes uint8 patches of one canvas, as one tensor or a list of tensors, in pinned memory or on the card, copied straight into the graph's input. Any batch size works: up to 32 pictures take at most two graph replays. Call it from one thread only.
- `canvases()` returns the 7 grids, as (rows, cols).

## CLI

```bash
fast-embeddinggemma embed ./photos --out vectors.npy  # every picture under the folder; the paths go to vectors.txt
fast-embeddinggemma bench ./pictures                  # model-only img/s at 1, 8 and 32 pictures per call
fast-embeddinggemma serve --port 8100                 # the HTTP service (needs the [server] extra)
```

## HTTP service

- `POST /embed {model, inputs: [{id, path} | {id, text}], query}` returns `{model, dims, vectors: [{id, vec}], failed: [{id, error}]}`. `vec` is a base64 little-endian float16 unit vector.
- Error codes: `409` if the request names another model; `503` if free video memory is under the floor (not checked for `query` requests); `422` for a malformed input.
- `GET /health` returns `{model, dims, freeVramMb, queue: {images, texts}, vram: "ok" | "low"}`.

How it handles work:
- Pictures are prepared on CPU threads, then batched across requests on one GPU thread. Each batch takes the canvas whose oldest picture has waited longest, with no waiting window.
- Text is always a search query. It gets the query prompt and runs through the engine's own text tower, one CUDA graph per length bucket, about 2 ms per query (the Hugging Face text model took 25 ms uncompiled). No second copy of the model is loaded: the query graphs share the engine's memory and add about 4 MiB. Queries over 512 tokens are cut at 512.
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

## Limits

- 140 soft tokens only. The 7 canvases cover about 96% of pictures with the grid the stock processor would have picked.
- The INT8 tile sizes were tuned on an RTX 3090 (sm_86). Other cards run, but may want other tiles.

## Code

```
src/fast_embeddinggemma/
  __init__.py        load, embed_images, embed_texts, process, embed, canvases
  preprocessing.py   any picture -> canvas + uint8 patches (CPU)
  server.py          the HTTP service
  cli.py, bench.py   fast-embeddinggemma embed | bench | serve
  modeling/          the network
    canvas.py        the 7 canvases, which one a picture goes on, their position and RoPE tables
    weights.py       reading the checkpoint, one layer reader for both towers
    norm.py          RMS norm, the fused add + norm kernel, bf16 rounding
    attention.py     the fused q/k norm + RoPE kernel, attention
    mlp.py           the fused GELU x up kernel
    int8.py          INT8 mode: weight quantization, its kernels, the INT8 vision layer
    vision.py        patches -> 16 layers -> 3x3 pooling -> soft tokens
    text.py          soft tokens -> 24-layer text tower -> the vector
    engine.py        load (weights + CUDA graphs) and embed
    queries.py       search queries through the engine's text tower, one graph per length bucket
```

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
