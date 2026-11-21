"""preprocessing.process against the locked v2 preparation (a native library plus its Pillow fallback), byte for byte: the canvas and the
patches of every picture in prep_reference.json. The seeded synthetic pictures run everywhere; library pictures are listed only
by the sha256 of their file (no picture or path is in the repo) and run when FEG_PICTURES names a folder holding them."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fast_embeddinggemma import preprocessing

REFERENCE = json.loads(Path(__file__).with_name("prep_reference.json").read_text())


def matches(src, ref):
    grid, patches = preprocessing.process(src)
    return list(grid) == ref["canvas"] and hashlib.sha256(patches.numpy().tobytes()).hexdigest() == ref["patches"]


def test_synthetic(pictures):
    changed = [n for n, r in REFERENCE["synthetic"].items() if hashlib.sha256(pictures[n]).hexdigest() != r["file"]]
    assert not changed, f"this Pillow encodes {changed} differently from the one the reference was made with"
    assert [n for n, r in REFERENCE["synthetic"].items() if not matches(pictures[n], r)] == []


def test_library(library):
    missing = [r["file"][:12] for r in REFERENCE["library"] if r["file"] not in library]
    assert not missing, f"{len(missing)} reference pictures are not in FEG_PICTURES"
    assert [library[r["file"]].name for r in REFERENCE["library"] if not matches(library[r["file"]], r)] == []


def test_types(pictures, tmp_path):
    """The same picture as a path, bytes, a PIL image and a uint8 RGB array gives the same canvas and the same patches."""
    data = pictures["webp_320x256"]
    path = tmp_path / "p.webp"
    path.write_bytes(data)
    rgb = Image.open(io.BytesIO(data)).convert("RGB")
    (grid, patches), *rest = [preprocessing.process(x) for x in (path, data, rgb, np.asarray(rgb))]
    assert all(g == grid and torch.equal(p, patches) for g, p in rest)
