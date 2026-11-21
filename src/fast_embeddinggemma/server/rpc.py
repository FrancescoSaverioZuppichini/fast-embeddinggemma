"""The gRPC service (embed.proto) on the same batcher as HTTP: Embed takes picture bytes, a path or a text per input; Health."""

import grpc

from . import batcher, embed_pb2, embed_pb2_grpc


def to_item(i: embed_pb2.Input) -> batcher.Item:
    kind = i.WhichOneof("source") or ""
    return i.id, kind, getattr(i, kind) if kind else ""


class Embedder(embed_pb2_grpc.EmbedderServicer):
    def __init__(self, state: dict) -> None:
        self.state = state

    async def Embed(
        self, request: embed_pb2.EmbedRequest, context: grpc.aio.ServicerContext
    ) -> embed_pb2.EmbedResponse:
        tag, dims = self.state["settings"].tag, self.state["settings"].dim
        if request.model != tag:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"this service runs {tag}, not {request.model}")
        try:
            vectors, failed = await batcher.embed_inputs(
                self.state, [to_item(i) for i in request.inputs], request.query
            )
        except batcher.Busy as error:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))
        return embed_pb2.EmbedResponse(
            model=tag,
            dims=dims,
            vectors=[embed_pb2.Vector(id=i, vec=v) for i, v in vectors],
            failed=[embed_pb2.Failure(id=i, error=e) for i, e in failed],
        )

    async def Health(
        self, request: embed_pb2.HealthRequest, context: grpc.aio.ServicerContext
    ) -> embed_pb2.HealthResponse:
        h = batcher.health(self.state)
        return embed_pb2.HealthResponse(
            model=h["model"],
            dims=h["dims"],
            free_vram_mb=h["freeVramMb"],
            queued_images=h["queue"]["images"],
            queued_texts=h["queue"]["texts"],
            vram=h["vram"],
        )


async def serve(state: dict) -> grpc.aio.Server:
    settings = state["settings"]
    # [Agent] gRPC refuses messages over 4 MB by default. Room for one full engine batch of pictures at the size limit,
    # capped at 1 GiB, since a whole message sits in memory before the handler sees it.
    limit = min(settings.max_upload_mb * settings.max_batch, 1024) << 20
    server = grpc.aio.server(options=[("grpc.max_receive_message_length", limit)])
    embed_pb2_grpc.add_EmbedderServicer_to_server(Embedder(state), server)
    server.add_insecure_port(f"{settings.host}:{settings.grpc_port}")
    await server.start()
    return server
