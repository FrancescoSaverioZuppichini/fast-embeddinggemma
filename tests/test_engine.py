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
    match, at cosine 0.998 or more (batch 1 against batch 32 sits near 0.9998), so a vector handed to the wrong picture fails."""
    items = list(library.values())[:40]
    together = feg.embed_images(engine, items)
    alone = np.concatenate([feg.embed_images(engine, [x]) for x in items])
    assert np.einsum("ij,ij->i", together, alone).min() > 0.998
    assert ((together @ alone.T).argmax(1) == np.arange(len(items))).all()
