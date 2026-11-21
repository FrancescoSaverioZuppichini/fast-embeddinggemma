"""fast-embeddinggemma serve | embed | bench. Each subcommand imports its own module, so `embed` and `bench` need no server extra."""

import argparse
from pathlib import Path

IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def list_pictures(folder: str) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGES)


def serve(args: argparse.Namespace) -> None:
    import logging

    import uvicorn

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

    run(list_pictures(args.folder)[: args.n], args.precision, tuple(args.sizes), args.threads)


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
    benchmark = commands.add_parser("bench", help="model-only img/s on the pictures of a folder")
    benchmark.add_argument("folder")
    benchmark.add_argument("--precision", default="int8", choices=("int8", "bf16"))
    benchmark.add_argument("--n", type=int, default=256, help="pictures: the first n by name")
    benchmark.add_argument("--sizes", type=int, nargs="+", default=[1, 8, 32], help="pictures per call")
    benchmark.add_argument("--threads", type=int, default=8, help="CPU threads that prepare the pictures")
    benchmark.set_defaults(run=bench)
    args = parser.parse_args(argv)
    args.run(args)
