# The HTTP service on an NVIDIA card: docker run --gpus all (see the README).
FROM ubuntu:24.04

# [Agent] The CUDA runtime comes inside the torch wheels and the driver from the host (--gpus all). Triton builds its small
# launcher with the system C compiler on first use, so gcc is the only system package needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --extra server --no-cache

# [Agent] Weights and compiled kernels go to /cache: mount a volume there so a restart neither downloads nor recompiles.
ENV PATH=/app/.venv/bin:$PATH HF_HOME=/cache/hf TRITON_CACHE_DIR=/cache/triton EMBED_HOST=0.0.0.0 EMBED_PORT=8100
VOLUME /cache
EXPOSE 8100
CMD ["fast-embeddinggemma", "serve"]
