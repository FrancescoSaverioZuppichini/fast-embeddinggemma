"""The service's settings: EMBED_* environment variables, typed and checked once at start-up."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBED_")

    model: str = Field("google/embeddinggemma-2", description="the name requests must send; vectors are tagged with it")
    precision: Literal["int8", "bf16"] = "int8"
    dim: Literal[768, 512, 256, 128] = 768
    host: str = "127.0.0.1"
    port: int = 8100
    grpc_port: int = Field(8101, description="0 turns gRPC off")
    path_roots: str = Field(
        "", description="':'-separated folders whose files may be sent by path; empty = any path, localhost only"
    )
    max_batch: Literal[32, 48, 64, 96, 128] = Field(32, description="most pictures one engine call takes")
    max_queued: int = Field(1024, description="pictures waiting before a request gets 503")
    max_upload_mb: int = 32
    min_free_mb: int = Field(900, description="free card memory under which picture requests get 503")
    threads: int = Field(8, description="CPU threads preparing pictures")

    def allows(self, path: str) -> bool:
        """May the service read this file? Under one of path_roots, or anywhere while it listens on localhost only."""
        if not self.path_roots:
            return self.host in LOOPBACK
        resolved = Path(path).resolve()
        return any(resolved.is_relative_to(Path(root).resolve()) for root in self.path_roots.split(":") if root)


@lru_cache
def get_settings() -> Settings:
    return Settings()
