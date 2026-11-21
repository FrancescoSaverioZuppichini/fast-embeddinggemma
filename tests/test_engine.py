"""The engine on the card (marked gpu; the weights must be in the Hugging Face cache): unit vectors of the right shape, and
embed_images hands each picture its own vector, in input order, whatever canvases and batch sizes it is cut into."""

import numpy as np
import pytest

import fast_embeddinggemma as feg

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def engine():
    return feg.load()


def test_vectors(engine, pictures):
    v = feg.embed_images(engine, list(pictures.values()))
    assert v.shape == (len(pictures), 768) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-3)


def test_order(engine, library):
    """Library pictures on all 7 canvases, one call against one call per picture: each row must be its own picture's best
    match, at cosine 0.99 or more. INT8 vectors move a little with the batch: bf16 rounding differs between batch sizes and
    flips INT8 roundings (batch 1 against 40 measured down to 0.9965 on library thumbnails), so the best match is the real check."""
    items = list(library.values())[:40]
    together = feg.embed_images(engine, items)
    alone = np.concatenate([feg.embed_images(engine, [x]) for x in items])
    assert np.einsum("ij,ij->i", together, alone).min() > 0.99
    assert ((together @ alone.T).argmax(1) == np.arange(len(items))).all()


def test_dim(engine, pictures):
    v = feg.embed_images(engine, list(pictures.values()), dim=256)
    assert v.shape == (len(pictures), 256) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-3)


def test_bad_picture(engine, pictures):
    with pytest.raises(ValueError, match="picture 1"):
        feg.embed_images(engine, [pictures["png_300x300"], b"not a picture"])
