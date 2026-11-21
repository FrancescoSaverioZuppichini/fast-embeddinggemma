"""Any picture (path, bytes, PIL image, uint8 RGB array) -> uint8 patches on one of the 7 canvases: Pillow (with the JPEG draft
scale-down for files and bytes) + EXIF turn + RGB + Lanczos thumbnail to 1,135 px, torchvision's uint8 bicubic antialiased squash
onto the canvas nearest in aspect ratio, then 16 x 16 patches in raster order (x fastest). The same pixels as the reference."""

import io
import os

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image, ImageOps

from .modeling.canvas import Grid, pick_canvas
from .modeling.vision import CFG as VISION

Picture = str | os.PathLike | bytes | Image.Image | np.ndarray
# [Agent] Twice the side 140 tokens need (48 px a token: int(2 * 48 * sqrt(140))): the first, cheap shrink of a big original.
SIDE = 1135


def decode(image: Picture) -> Image.Image:
    """Any picture -> RGB no larger than SIDE: the JPEG draft scale-down (files and bytes only: a PIL image is already decoded),
    the EXIF turn, RGB, a Lanczos thumbnail. The caller's image is never changed."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image).convert("RGB")
    else:
        with Image.open(io.BytesIO(image) if isinstance(image, bytes) else image) as opened:
            opened.draft("RGB", (SIDE, SIDE))
            image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((SIDE, SIDE), Image.Resampling.LANCZOS)
    return image


def process(image: Picture) -> tuple[Grid, torch.Tensor]:
    """Any picture -> ((rows, cols), uint8 patches (9 rows cols, 768))."""
    image = decode(image)
    rows, cols = pick_canvas(image.width, image.height)
    # [Agent] A canvas cell is pool x pool patches of patch x patch pixels: 3 x 16 = 48 px.
    p, n = VISION.patch, VISION.pool
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    pixels = tvF.resize(
        pixels, [p * n * rows, p * n * cols], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True
    )
    patches = pixels.reshape(3, n * rows, p, n * cols, p).permute(1, 3, 2, 4, 0)
    return (rows, cols), patches.reshape(n * n * rows * cols, 3 * p * p)
