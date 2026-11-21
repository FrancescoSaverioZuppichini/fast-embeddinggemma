"""Any picture (path, bytes, PIL image, uint8 RGB array) -> uint8 patches on one of the 7 canvases, exactly what the stock sidecar feeds the model at 140
tokens: Pillow draft + EXIF turn + RGB + Lanczos thumbnail to 1,135 px, torchvision's uint8 bicubic antialias squash onto the
canvas nearest in log aspect ratio, then 16 x 16 patches in the processor's order (raster, x fastest)."""

import io

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image, ImageOps

from .modeling.canvas import pick_canvas

SIDE = 1135  # the stock decode side at 140 tokens: int(2 * 48 * sqrt(140)), twice the side a token budget needs


def decode(image):
    """Any picture -> RGB no larger than SIDE, the stock sidecar's decode: the JPEG draft scale-down (files and bytes only:
    a PIL image is already decoded), the EXIF turn, RGB, a Lanczos thumbnail. The caller's image is never changed."""
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


def process(image):
    """Any picture -> ((rows, cols), uint8 patches (9 rows cols, 768))."""
    image = decode(image)
    rows, cols = pick_canvas(image.width, image.height)
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    pixels = tvF.resize(pixels, [48 * rows, 48 * cols], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    return (rows, cols), pixels.reshape(3, 3 * rows, 16, 3 * cols, 16).permute(1, 3, 2, 4, 0).reshape(
        9 * rows * cols, 768
    )
