"""A search query over 512 tokens runs (sliding-window mask, eager) and gives the same vector alone or beside short ones."""

import numpy as np
import pytest

import fast_embeddinggemma as feg

pytestmark = pytest.mark.gpu


def test_long_query():
    model = feg.load()
    long = "receipts from the dentist and the car insurance, " * 120
    texts = ["a cat on a sofa", long, "sunset over the lake"]
    together = feg.embed_texts(model, texts)
    alone = np.concatenate([feg.embed_texts(model, [t]) for t in texts])
    assert np.allclose(np.linalg.norm(together, axis=1), 1, atol=1e-3)
    assert np.einsum("ij,ij->i", together, alone).min() > 0.9999
