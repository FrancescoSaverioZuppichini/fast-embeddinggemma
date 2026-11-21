"""The HTTP contract, as Pydantic models (they are the OpenAPI page at /docs)."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Input(BaseModel):
    id: str = Field(description="echoed back with its vector or its error")
    path: str | None = Field(None, description="a picture file on the server's disk (see EMBED_PATH_ROOTS)")
    text: str | None = Field(None, description="a search query; the query prompt is added")

    @model_validator(mode="after")
    def one_source(self) -> "Input":
        if (self.path is None) == (self.text is None):
            raise ValueError("an input has exactly one of path or text")
        return self


class EmbedRequest(BaseModel):
    model: str = Field(description="must be the tag /health reports: checkpoint@precision")
    inputs: list[Input]
    query: bool = Field(False, description="a search: skips the free-memory floor kept for background batches")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "google/embeddinggemma-2@int8",
                    "inputs": [{"id": "p1", "path": "/photos/beach.jpg"}, {"id": "q1", "text": "red dress on a beach"}],
                }
            ]
        }
    }


class Vector(BaseModel):
    id: str
    vec: str = Field(description="base64 of `dims` little-endian float16 values, unit length: cosine = dot product")


class Failure(BaseModel):
    id: str
    error: str


class EmbedResponse(BaseModel):
    model: str
    dims: int
    vectors: list[Vector]
    failed: list[Failure] = Field(description="items that could not be embedded, each with its reason")


class Queue(BaseModel):
    images: int
    texts: int


class Health(BaseModel):
    model: str = Field(description="checkpoint@precision; requests send it")
    dims: int
    freeVramMb: int
    queue: Queue
    vram: Literal["ok", "low"]
