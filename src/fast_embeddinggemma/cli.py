"""fast-embeddinggemma serve | bench. Each subcommand imports its own module, so `bench` needs no server extra."""

import argparse


def serve(args):
    import uvicorn

    uvicorn.run("fast_embeddinggemma.server:app", host=args.host, port=args.port)


def bench(args):
    from .bench import run

    run(args.folder, args.precision, args.n, tuple(args.sizes), args.threads)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fast-embeddinggemma")
    commands = parser.add_subparsers(required=True)
    p = commands.add_parser("serve", help="the HTTP service (settings: EMBED_* environment variables)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.set_defaults(run=serve)
    p = commands.add_parser("bench", help="model-only img/s on a folder of pictures")
    p.add_argument("folder")
    p.add_argument("--precision", default="int8", choices=("int8", "bf16"))
    p.add_argument("--n", type=int, default=256, help="pictures: the first n files by name")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 8, 32], help="pictures per call")
    p.add_argument("--threads", type=int, default=8, help="CPU threads that prepare the pictures")
    p.set_defaults(run=bench)
    args = parser.parse_args(argv)
    args.run(args)
