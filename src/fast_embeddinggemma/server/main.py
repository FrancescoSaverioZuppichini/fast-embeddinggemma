"""The HTTP service: POST /embed (paths or texts), POST /embed/files (uploaded pictures), GET /health, the OpenAPI page at
/docs; the gRPC service (rpc.py) runs beside it on the same batcher. Settings: EMBED_* environment variables (config.py)."""

import base64
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile

from .. import __version__
from . import batcher, rpc
from .config import Settings
from .schemas import EmbedRequest, EmbedResponse, Failure, Health, Vector

log = logging.getLogger("fast_embeddinggemma")
BUSY = {503: {"description": "too many pictures in flight, or the card is short of memory: retry later"}}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = Settings()
    app.state.batcher = batcher.start(settings)
    grpc_server = await rpc.serve(app.state.batcher) if settings.grpc_port else None
    log.info("server.start.completed", extra={"port": settings.port, "grpc_port": settings.grpc_port})
    yield
    if grpc_server:
        await grpc_server.stop(grace=5)
    batcher.stop(app.state.batcher)


app = FastAPI(
    title="fast-embeddinggemma",
    version=__version__,
    summary="EmbeddingGemma 2 picture and search-query embeddings on an NVIDIA card: one 768-d space, unit vectors.",
    lifespan=lifespan,
)


# [Agent] async so FastAPI calls it on the event loop; a plain def dependency is sent to the thread pool on every request.
async def get_batcher(request: Request) -> dict:
    return request.app.state.batcher


State = Annotated[dict, Depends(get_batcher)]


async def respond(state: dict, items: list[batcher.Item], query: bool) -> EmbedResponse:
    try:
        vectors, failed = await batcher.embed_inputs(state, items, query)
    except batcher.Busy as error:
        raise HTTPException(503, str(error)) from error
    settings = state["settings"]
    return EmbedResponse(
        model=settings.tag,
        dims=settings.dim,
        vectors=[Vector(id=i, vec=base64.b64encode(v).decode()) for i, v in vectors],
        failed=[Failure(id=i, error=e) for i, e in failed],
    )


@app.post("/embed", responses={409: {"description": "the request names another model"}, **BUSY})
async def embed(request: EmbedRequest, state: State) -> EmbedResponse:
    """Pictures by path on the server's disk, and search queries, in one call; each input gets its vector or its error."""
    tag = state["settings"].tag
    if request.model != tag:
        raise HTTPException(409, f"this service runs {tag}, not {request.model}")
    items: list[batcher.Item] = [
        (i.id, "path", i.path) if i.path is not None else (i.id, "text", i.text or "") for i in request.inputs
    ]
    return await respond(state, items, request.query)


@app.post("/embed/files", responses={413: {"description": "too many files, or one is too big"}, **BUSY})
async def embed_files(
    files: Annotated[list[UploadFile], File(description="one or more pictures")],
    state: State,
    ids: Annotated[list[str] | None, Form(description="one id per file; default: the file name")] = None,
) -> EmbedResponse:
    """Uploaded pictures (multipart/form-data), for clients on another machine."""
    settings: Settings = state["settings"]
    if ids and len(ids) != len(files):
        raise HTTPException(422, "send one id per file")
    if len(files) > settings.max_queued:
        raise HTTPException(413, f"send at most {settings.max_queued} files")
    # [Agent] Sizes are checked before any file is read into memory; the multipart parser has already spooled them to disk.
    for upload in files:
        if (upload.size or 0) > settings.max_upload_mb << 20:
            raise HTTPException(413, f"{upload.filename} is over {settings.max_upload_mb} MB")
    items: list[batcher.Item] = [
        (ids[k] if ids else upload.filename or str(k), "data", await upload.read()) for k, upload in enumerate(files)
    ]
    return await respond(state, items, query=False)


@app.get("/health")
async def health(state: State) -> Health:
    """What the service runs, how many pictures are in flight, and whether the card has memory for picture batches."""
    return Health(**batcher.health(state))
