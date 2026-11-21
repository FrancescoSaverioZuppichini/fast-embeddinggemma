from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class Input(_message.Message):
    __slots__ = ("id", "data", "path", "text")
    ID_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    id: str
    data: bytes
    path: str
    text: str
    def __init__(
        self,
        id: str | None = ...,
        data: bytes | None = ...,
        path: str | None = ...,
        text: str | None = ...,
    ) -> None: ...

class EmbedRequest(_message.Message):
    __slots__ = ("model", "inputs", "query")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    model: str
    inputs: _containers.RepeatedCompositeFieldContainer[Input]
    query: bool
    def __init__(
        self,
        model: str | None = ...,
        inputs: _Iterable[Input | _Mapping] | None = ...,
        query: bool | None = ...,
    ) -> None: ...

class Vector(_message.Message):
    __slots__ = ("id", "vec")
    ID_FIELD_NUMBER: _ClassVar[int]
    VEC_FIELD_NUMBER: _ClassVar[int]
    id: str
    vec: bytes
    def __init__(self, id: str | None = ..., vec: bytes | None = ...) -> None: ...

class Failure(_message.Message):
    __slots__ = ("id", "error")
    ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    id: str
    error: str
    def __init__(self, id: str | None = ..., error: str | None = ...) -> None: ...

class EmbedResponse(_message.Message):
    __slots__ = ("model", "dims", "vectors", "failed")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    DIMS_FIELD_NUMBER: _ClassVar[int]
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    FAILED_FIELD_NUMBER: _ClassVar[int]
    model: str
    dims: int
    vectors: _containers.RepeatedCompositeFieldContainer[Vector]
    failed: _containers.RepeatedCompositeFieldContainer[Failure]
    def __init__(
        self,
        model: str | None = ...,
        dims: int | None = ...,
        vectors: _Iterable[Vector | _Mapping] | None = ...,
        failed: _Iterable[Failure | _Mapping] | None = ...,
    ) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("model", "dims", "free_vram_mb", "queued_images", "queued_texts", "vram")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    DIMS_FIELD_NUMBER: _ClassVar[int]
    FREE_VRAM_MB_FIELD_NUMBER: _ClassVar[int]
    QUEUED_IMAGES_FIELD_NUMBER: _ClassVar[int]
    QUEUED_TEXTS_FIELD_NUMBER: _ClassVar[int]
    VRAM_FIELD_NUMBER: _ClassVar[int]
    model: str
    dims: int
    free_vram_mb: int
    queued_images: int
    queued_texts: int
    vram: str
    def __init__(
        self,
        model: str | None = ...,
        dims: int | None = ...,
        free_vram_mb: int | None = ...,
        queued_images: int | None = ...,
        queued_texts: int | None = ...,
        vram: str | None = ...,
    ) -> None: ...
