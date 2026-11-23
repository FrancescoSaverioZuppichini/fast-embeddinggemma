"""fast-embeddinggemma serve | embed | bench | eval. Each subcommand imports its own module, so `embed` and `bench` need no
server extra; `eval` and a `bench` without a folder need the eval extra (the dataset)."""

import argparse
from pathlib import Path

IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def list_pictures(folder: str) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGES)


def serve(args: argparse.Namespace) -> None:
    import logging
    import os

    import uvicorn

    # [Agent] Expandable segments let PyTorch's allocator grow mappings instead of rounding up many fixed blocks: the
    # service holds 3.1 GB of card memory instead of 4.5, and every benchmark ran this way. PyTorch reads it when CUDA
    # first allocates, so it is set before anything touches the card; a value already in the environment wins.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from .server.config import Settings

    # [Agent] The service's events go through stdlib logging; without a handler they would print nothing.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings()
    uvicorn.run("fast_embeddinggemma.server.main:app", host=settings.host, port=settings.port)


def embed(args: argparse.Namespace) -> None:
    import numpy as np

    from . import embed_images, load

    paths = list_pictures(args.folder)
    vectors = embed_images(load(args.precision), paths, dim=args.dim)
    np.save(args.out, vectors)
    names = Path(args.out).with_suffix(".txt")
    names.write_text("".join(f"{p}\n" for p in paths))
    print(f"{len(paths)} pictures -> {args.out} {vectors.shape}, paths in {names}")


def bench(args: argparse.Namespace) -> None:
    from .bench import run

    if args.folder:
        pictures = list_pictures(args.folder)
    else:
        from .quality import load_rows

        pictures = list(load_rows(args.revision, args.data)["test"]["thumb"])
    run(pictures[: args.n], args.precision, tuple(args.sizes), args.threads)


def evaluate(args: argparse.Namespace) -> None:
    import os

    # [Agent] As for `serve`: the allocator grows mappings instead of rounding up fixed blocks, so the eval fits beside
    # another process on the card (3.1 GB instead of 4.5). Set before torch touches CUDA; the environment wins.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from .quality import run

    run(args.precision, args.dim, args.revision, args.data, args.out)


def convert(args: argparse.Namespace) -> None:
    from .modeling.convert import convert as run

    print(f"weights -> {run(args.out, args.checkpoint)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fast-embeddinggemma")
    commands = parser.add_subparsers(required=True)
    serving = commands.add_parser("serve", help="the HTTP + gRPC service (settings: EMBED_* environment variables)")
    serving.set_defaults(run=serve)
    embedding = commands.add_parser("embed", help="every picture under a folder -> vectors.npy (+ the paths in .txt)")
    embedding.add_argument("folder")
    embedding.add_argument("--out", default="vectors.npy")
    embedding.add_argument("--precision", default="int8", choices=("int8", "bf16"))
    embedding.add_argument("--dim", type=int, default=768, choices=(768, 512, 256, 128))
    embedding.set_defaults(run=embed)
    benchmark = commands.add_parser(
        "bench", help="model-only img/s on the pictures of a folder, or on aspect-retrieval-5k's thumbnails"
    )
    benchmark.add_argument("folder", nargs="?")
    benchmark.add_argument("--precision", default="int8", choices=("int8", "bf16"))
    benchmark.add_argument("--n", type=int, help="pictures: the first n (by name in a folder), default all")
    benchmark.add_argument("--sizes", type=int, nargs="+", default=[1, 8, 32], help="pictures per call")
    benchmark.add_argument("--threads", type=int, default=8, help="CPU threads that prepare the pictures")
    benchmark.set_defaults(run=bench)
    evaluating = commands.add_parser(
        "eval", help="search quality on aspect-retrieval-5k: R@1/5/10 and MRR per query kind"
    )
    evaluating.add_argument("--precision", default="int8", choices=("int8", "bf16"))
    evaluating.add_argument("--dim", type=int, default=768, choices=(768, 512, 256, 128))
    evaluating.add_argument("--out", default="quality.json")
    evaluating.set_defaults(run=evaluate)
    for command in (benchmark, evaluating):
        command.add_argument("--revision", default="v1.0", help="the dataset version on the Hub")
        command.add_argument("--data", help="a local copy saved with save_to_disk, instead of the Hub")
    converting = commands.add_parser("convert", help="Google's checkpoint -> our weights + tokenizer in a folder")
    converting.add_argument("out", help="folder for model.safetensors and tokenizer.json")
    converting.add_argument(
        "--checkpoint", help="a local file in the layout of google/embeddinggemma-2's model.safetensors"
    )
    converting.set_defaults(run=convert)
    args = parser.parse_args(argv)
    args.run(args)
