"""Picture (path or bytes) -> uint8 patches on one of the 7 canvases, exactly what the stock sidecar feeds the model at 140
tokens: Pillow draft + EXIF turn + RGB + Lanczos thumbnail to 1,135 px, torchvision's uint8 bicubic antialias squash onto the
canvas nearest in log aspect ratio, then 16 x 16 patches in the processor's order (raster, x fastest)."""

import io
import math

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image, ImageOps

from .model import canvases

SIDE = 1135  # the stock decode side at 140 tokens: int(2 * 48 * sqrt(140)), twice the side a token budget needs


def decode(src):
    """The stock sidecar's decode: the JPEG draft scale-down, the EXIF turn, RGB, a Lanczos thumbnail to SIDE."""
    with Image.open(io.BytesIO(src) if isinstance(src, bytes) else src) as image:
        image.draft("RGB", (SIDE, SIDE))
        image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((SIDE, SIDE), Image.Resampling.LANCZOS)
    return image


def pick_canvas(width, height):
    # [Agent] Nearest in log aspect ratio, ties to more cells and then to the smaller (rows, cols): the locked native rule.
    r = math.log(width / height)
    return min(canvases(), key=lambda g: (abs(math.log(g[1] / g[0]) - r), -g[0] * g[1], g))


def read(src):
    """Path or bytes -> ((rows, cols), uint8 patches (9 rows cols, 768))."""
    image = decode(src)
    rows, cols = pick_canvas(image.width, image.height)
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    pixels = tvF.resize(pixels, [48 * rows, 48 * cols], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    return (rows, cols), pixels.reshape(3, 3 * rows, 16, 3 * cols, 16).permute(1, 3, 2, 4, 0).reshape(
        9 * rows * cols, 768
    )
