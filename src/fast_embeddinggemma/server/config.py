"""The service's settings: EMBED_* environment variables, typed and checked once at start-up."""

from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..modeling.convert import GOOGLE


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBED_")

    precision: Literal["int8", "bf16"] = "int8"
    dim: Literal[768, 512, 256, 128] = 768
    host: str = "127.0.0.1"
    port: int = 8100
    grpc_port: int = Field(8101, description="0 turns gRPC off")
    path_roots: str = Field("", description="':'-separated folders whose files may be sent by path; empty = no paths")
    max_batch: Literal[32, 48, 64, 96, 128] = Field(32, description="most pictures one engine call takes")
    max_queued: int = Field(1024, description="pictures in flight (decoding or waiting) before a request gets 503")
    max_upload_mb: int = Field(32, description="largest picture sent as bytes")
    min_free_mb: int = Field(900, description="free card memory under which picture requests get 503")
    threads: int = Field(8, description="CPU threads preparing pictures")

    @property
    def tag(self) -> str:
        """What the vectors are: the checkpoint and its precision. Requests must name it; a new tag is a new vector space."""
        return f"{GOOGLE}@{self.precision}"

    @cached_property
    def roots(self) -> list[Path]:
        return [Path(root).resolve() for root in self.path_roots.split(":") if root]

    def allows(self, path: Path) -> bool:
        """May the service read this resolved path? Only under one of path_roots."""
        return any(path.is_relative_to(root) for root in self.roots)
