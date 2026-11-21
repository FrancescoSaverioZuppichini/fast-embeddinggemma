"""The service: settings and the input schema on the CPU; on the card (marked gpu) the real app through its lifespan, HTTP and gRPC,
with pictures sent by path, as uploaded bytes and over gRPC, against the Python API's vectors."""

import base64
import socket

import numpy as np
import pytest
from pydantic import ValidationError

from fast_embeddinggemma.server.config import Settings
from fast_embeddinggemma.server.schemas import Input


def test_settings_reject_unknown_dim(monkeypatch):
    monkeypatch.setenv("EMBED_DIM", "100")
    with pytest.raises(ValidationError):
        Settings()


def test_paths_allowed(tmp_path):
    inside, outside = tmp_path / "a.jpg", "/etc/passwd"
    assert Settings(path_roots=str(tmp_path)).allows(str(inside))
    assert not Settings(path_roots=str(tmp_path)).allows(outside)
    assert not Settings(path_roots=str(tmp_path)).allows(str(tmp_path / ".." / "x.jpg"))
    assert Settings(host="127.0.0.1").allows(outside)
    assert not Settings(host="0.0.0.0").allows(outside)


def test_input_has_one_source():
    with pytest.raises(ValidationError):
        Input(id="x")
    with pytest.raises(ValidationError):
        Input(id="x", path="/a.jpg", text="b")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.gpu
def test_service(tmp_path, pictures, monkeypatch):
    import grpc
    from fastapi.testclient import TestClient

    import fast_embeddinggemma as feg
    from fast_embeddinggemma.server import embed_pb2, embed_pb2_grpc
    from fast_embeddinggemma.server.config import get_settings
    from fast_embeddinggemma.server.main import app

    port = free_port()
    monkeypatch.setenv("EMBED_PATH_ROOTS", str(tmp_path))
    monkeypatch.setenv("EMBED_GRPC_PORT", str(port))
    get_settings.cache_clear()
    names = ["webp_320x256", "jpeg_1600x1200", "png_300x300"]
    for n in names:
        (tmp_path / n).write_bytes(pictures[n])

    def vectors(answer: dict) -> np.ndarray:
        return np.stack(
            [np.frombuffer(base64.b64decode(v["vec"]), "<f2").astype(np.float32) for v in answer["vectors"]]
        )

    with TestClient(app) as client:
        model = client.get("/health").json()["model"]
        inputs = [{"id": n, "path": str(tmp_path / n)} for n in names] + [{"id": "q", "text": "a cat on a sofa"}]
        by_path = client.post("/embed", json={"model": model, "inputs": inputs}).json()
        assert [v["id"] for v in by_path["vectors"]] == [*names, "q"] and by_path["failed"] == []
        files = [("files", (n, pictures[n])) for n in names]
        uploaded = client.post("/embed/files", files=files).json()
        outside = client.post("/embed", json={"model": model, "inputs": [{"id": "x", "path": "/etc/hostname"}]}).json()
        assert outside["failed"][0]["error"] == "path outside EMBED_PATH_ROOTS"
        assert client.post("/embed", json={"model": "other", "inputs": inputs}).status_code == 409
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            request = embed_pb2.EmbedRequest(
                model=model, inputs=[embed_pb2.Input(id=n, data=pictures[n]) for n in names]
            )
            answer = embed_pb2_grpc.EmbedderStub(channel).Embed(request)
        over_grpc = np.stack([np.frombuffer(v.vec, "<f2").astype(np.float32) for v in answer.vectors])
        engine = client.app.state.batcher["engine"]
        reference = feg.embed_images(engine, [pictures[n] for n in names])
    # [Agent] float16 transport costs ~1e-3; batching across requests moves INT8 vectors up to ~3e-3 (see test_order).
    for got in (vectors(by_path)[:3], vectors(uploaded), over_grpc):
        assert np.einsum("ij,ij->i", got, reference).min() > 0.99
