import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def make_pictures():
    """Seeded pictures that take every branch of the locked v2 preparation: WebP thumbnails on three canvases, JPEGs that
    need the draft scale-down and the Lanczos thumbnail, one that needs Pillow's reduce step, a PNG and an EXIF-turned JPEG
    (the last three went through the locked path's Pillow fallback). RandomState: its stream never changes across NumPy."""
    rng = np.random.RandomState(0)

    def picture(w, h):
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        f = rng.uniform(2, 12, 3)
        base = 127 + 100 * np.stack([np.sin(f[c] * (x / w + c * y / h) + c) for c in range(3)], -1)
        return Image.fromarray(np.clip(base + rng.normal(0, 12, (h, w, 3)), 0, 255).astype(np.uint8))

    def encode(image, fmt, **options):
        buf = io.BytesIO()
        image.save(buf, fmt, **options)
        return buf.getvalue()

    turned = Image.Exif()
    turned[0x0112] = 6
    return {
        "webp_320x256": encode(picture(320, 256), "WEBP", quality=80),
        "webp_180x320": encode(picture(180, 320), "WEBP", quality=80),
        "webp_320x180": encode(picture(320, 180), "WEBP", quality=80),
        "jpeg_1600x1200": encode(picture(1600, 1200), "JPEG", quality=90),
        "jpeg_3000x2250": encode(picture(3000, 2250), "JPEG", quality=90),
        "jpeg_4600x1150": encode(picture(4600, 1150), "JPEG", quality=90),
        "png_300x300": encode(picture(300, 300), "PNG"),
        "jpeg_exif6_256x320": encode(picture(256, 320), "JPEG", quality=90, exif=turned),
    }


@pytest.fixture(scope="session")
def pictures():
    return make_pictures()


@pytest.fixture(scope="session")
def library():
    """FEG_PICTURES: a folder of your own pictures, {sha256 of the file: path}; the tests that need it skip without it."""
    if not os.environ.get("FEG_PICTURES"):
        pytest.skip("FEG_PICTURES is not set")
    import hashlib

    paths = [p for p in sorted(Path(os.environ["FEG_PICTURES"]).iterdir()) if p.is_file()]
    return {hashlib.sha256(p.read_bytes()).hexdigest(): p for p in paths}
