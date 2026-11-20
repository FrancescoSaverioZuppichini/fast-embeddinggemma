"""Google's QAT EmbeddingGemma 2 text+vision LiteRT-LM bundle -> one safetensors checkpoint for this engine, under Hugging
Face names (the stock HF code loads it too). Every tensor is found by NAME and data flow in Google's graphs.
  vision products  <name>.int8 (out, in) + <name>.scale (out,) per channel; <name> = the dequantized fp32 copy
  vision act       vision_tower.encoder.layers.<i>.act_in (4: q|k|v, o, gate|up, down inputs) and .act_out (7: q k v o gate up
                   down outputs): the static per-tensor INT8 scales Google trained with
  text products    INT4 per channel, dequantized to fp32
  folded constants undone: patch weight x0.5 (Google feeds pixels in [0, 1]), PLE projection / 512, special rows / sqrt(512)
A test tool (it searches worse than this package's own INT8): fast-embeddinggemma convert-google <bundle.litertlm> <out>"""

import re

import flatbuffers
import numpy as np
import torch
from ai_edge_litert import schema_py_generated as S
from flatbuffers.table import Table
from safetensors.torch import save_file

CFG = dict(
    header=32,
    model=3,
    weights=7,
    vision=("tf_lite_vision_encoder", "vision_140"),
    adapter=("tf_lite_vision_adapter", "vision_adapter_140"),
    text=("tf_lite_text_encoder", "encoder_1x256"),
    embedder=("tf_lite_embedder", None),
    eoi=("tf_lite_end_of_vision", None),
    special=(2, 255999, 258882, 1),
    vision_proj={
        "q_einsum": "self_attn.q_proj",
        "k_einsum": "self_attn.k_proj",
        "v_einsum": "self_attn.v_proj",
        "attn_vec_einsum": "self_attn.o_proj",
        "gating_einsum1": "mlp.gate_proj",
        "gating_einsum2": "mlp.up_proj",
        "mlp/linear": "mlp.down_proj",
    },
    norms={
        "pre_attention_norm": "input_layernorm",
        "query_norm": "self_attn.q_norm",
        "key_norm": "self_attn.k_norm",
        "post_attention_norm": "post_attention_layernorm",
        "pre_ffw_norm": "pre_feedforward_layernorm",
        "post_ffw_norm": "post_feedforward_layernorm",
        "post_per_layer_input_norm": "ple_block.post_per_layer_input_norm",
    },
    act_in=("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"),
    act_out=(
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ),
)
DTYPES = {
    S.TensorType.FLOAT32: np.float32,
    S.TensorType.FLOAT16: np.float16,
    S.TensorType.INT8: np.int8,
    S.TensorType.INT32: np.int32,
}


def read_sections(raw):
    """{(model type, section kind): begin offset} from the bundle header (litertlm_header_schema.fbs: sections, key/values)."""

    def field(t, i):
        return t.Offset(4 + 2 * i)

    def vec(t, i):
        o = field(t, i)
        return [Table(t.Bytes, t.Indirect(t.Vector(o) + 4 * j)) for j in range(t.VectorLen(o))] if o else []

    def sub(t, i):
        o = field(t, i)
        return Table(t.Bytes, t.Indirect(t.Pos + o)) if o else None

    root = Table(raw, CFG["header"] + flatbuffers.encode.Get(flatbuffers.packer.uoffset, raw, CFG["header"]))
    out = {}
    for s in vec(sub(root, 1), 0):
        kv = {k.String(k.Pos + field(k, 0)).decode(): sub(k, 2) for k in vec(s, 0)}
        v = kv.get("model_type")
        kind = s.Get(flatbuffers.number_types.Uint8Flags, s.Pos + field(s, 3))
        begin = s.Get(flatbuffers.number_types.Uint64Flags, s.Pos + field(s, 1))
        end = s.Get(flatbuffers.number_types.Uint64Flags, s.Pos + field(s, 2))
        if v is not None:
            out[v.String(v.Pos + field(v, 0)).decode(), kind] = (begin, end)
    return out


def load_graph(raw, sections, which):
    """(model, subgraph, read): read(tensor index) -> fp32 array (dequantized), raw ints, scales; external weights resolved."""
    kind, graph = which
    b, e = sections[kind, CFG["model"]]
    w0 = sections.get((kind, CFG["weights"]), (0, 0))[0]
    m = S.ModelT.InitFromPackedBuf(raw[b:e], 0)
    g = m.subgraphs[0] if graph is None else next(x for x in m.subgraphs if x.name.decode() == graph)
    ext = {x.id: (x.offset, x.length) for x in (m.externalBuffers or [])}

    def read(i):
        t = g.tensors[i]
        if t.externalBuffer:
            off, n = ext[t.externalBuffer]
            data = raw[w0 + off : w0 + off + n]
        else:
            d = m.buffers[t.buffer].data
            data = b"" if d is None else d.tobytes()
        shape = [int(s) for s in t.shape]
        count = int(np.prod(shape)) if shape else 1
        ints = unpack_int4(data, count) if t.type == S.TensorType.INT4 else np.frombuffer(data, DTYPES[t.type], count)
        ints = ints.reshape(shape)
        q = t.quantization
        if q is None or q.scale is None or not len(q.scale) or t.type in (S.TensorType.FLOAT32, S.TensorType.FLOAT16):
            return ints.astype(np.float32), None, None
        scale = np.asarray(q.scale, np.float32)
        view = [1] * len(shape)
        view[q.quantizedDimension if scale.size > 1 else 0] = -1
        return ints.astype(np.float32) * scale.reshape(view if scale.size > 1 else [1]), ints, scale

    return m, g, read


def unpack_int4(data, n):
    # [Agent] TFLite INT4 packs two signed values per byte, low nibble first; each nibble is sign-extended from 4 bits.
    b = np.frombuffer(data, np.uint8)
    v = np.stack([b & 15, b >> 4], -1).reshape(-1)[:n].astype(np.int8)
    return np.where(v > 7, v - 16, v).astype(np.int8)


def opcode(m, op):
    c = m.operatorCodes[op.opcodeIndex]
    return max(c.builtinCode, c.deprecatedBuiltinCode)


def tname(g, i):
    return g.tensors[i].name.decode()


def constants(m, g, op):
    """Indices of the op inputs that hold data (weights, scales, tables)."""
    return [
        i
        for i in op.inputs
        if i >= 0 and (g.tensors[i].externalBuffer or m.buffers[g.tensors[i].buffer].data is not None)
    ]


def producers(g):
    return {o: op for op in g.operators for o in op.outputs}


def trace_back(m, g, prod, i, code):
    """Walk from tensor i up through single-path ops (reshape, transpose, ...) to the nearest op of this code."""
    while i in prod:
        op = prod[i]
        if opcode(m, op) == code:
            return op
        live = [x for x in op.inputs if x >= 0 and x not in constants(m, g, op)]
        if not live:
            return None
        i = live[0]
    return None


def find_up(m, g, prod, i, code):
    """Breadth-first up the data flow from tensor i (every non-constant input) to the nearest op of this code."""
    todo, seen = [i], set()
    while todo:
        i = todo.pop(0)
        if i in seen or i not in prod:
            continue
        seen.add(i)
        op = prod[i]
        if opcode(m, op) == code:
            return op
        todo += [x for x in op.inputs if x >= 0 and x not in constants(m, g, op)]
    return None


def layer_of(name):
    hit = re.search(r"layer_(\d+)/", name)
    return int(hit.group(1)) if hit else None


def norm_scales(m, g, read, prefix):
    """{hf name: scale} of every named norm op of a layer stack (the constant input of the op named after the norm)."""
    out = {}
    for op in g.operators:
        hit = re.search(r"layer_(\d+)/.*?/(\w+_norm)/composite\d*$", tname(g, op.outputs[0]))
        if hit and hit.group(2) in CFG["norms"]:
            out[f"{prefix}.layers.{hit.group(1)}.{CFG['norms'][hit.group(2)]}.weight"] = read(constants(m, g, op)[0])[0]
    return out


def map_vision(m, g, read):
    """Vision: INT8 products + scales, norms, static activation scales, patch weight (x0.5) and the two position tables."""
    p, out, fc = "vision_tower.encoder", {}, S.BuiltinOperator.FULLY_CONNECTED
    prod, act_in, act_out = producers(g), {}, {}
    for op in g.operators:
        if opcode(m, op) != fc:
            continue
        w = constants(m, g, op)[0]
        wname, k = tname(g, w).split(";")[0], layer_of(tname(g, w))
        proj = next((v for key, v in CFG["vision_proj"].items() if key in wname), None)
        if k is None or proj is None:
            continue
        deq, ints, scale = read(w)
        key = f"{p}.layers.{k}.{proj}.linear.weight"
        out[key], out[key + ".int8"], out[key + ".scale"] = deq, ints, scale
        q_in = g.tensors[op.inputs[0]].quantization
        act_in.setdefault(k, {})[proj] = float(q_in.scale[0])
        act_out.setdefault(k, {})[proj] = float(g.tensors[op.outputs[0]].quantization.scale[0])
    for k in act_in:
        out[f"{p}.layers.{k}.act_in"] = np.array([act_in[k][n] for n in CFG["act_in"]], np.float32)
        out[f"{p}.layers.{k}.act_out"] = np.array([act_out[k][n] for n in CFG["act_out"]], np.float32)
    out |= norm_scales(m, g, read, p)
    out |= map_entry(m, g, read, prod)
    return out


def map_entry(m, g, read, prod):
    """Patch weight (Google feeds pixels in [0, 1], HF 2 p - 1: same product with half the weight) and the x / y position
    tables, each identified by the coordinate its one-hot reads (the SLICE of positions_xy at index 0 = x, 1 = y)."""
    out, tables = {}, {}
    for op in g.operators:
        n = tname(g, op.outputs[0])
        if opcode(m, op) != S.BuiltinOperator.FULLY_CONNECTED or "entry/" not in n:
            continue
        w = read(constants(m, g, op)[0])[0]
        if "input_projection" in n:
            out["vision_tower.patch_embedder.input_proj.weight"] = 0.5 * w
            continue
        sl = find_up(m, g, prod, op.inputs[0], S.BuiltinOperator.SLICE)
        axis = int(read(constants(m, g, sl)[0])[0].reshape(-1)[-1])
        tables[axis] = w.T if w.shape[0] == 768 else w
    out["vision_tower.patch_embedder.position_embedding_table"] = np.stack([tables[0], tables[1]])
    return out


def map_adapter(m, g, read):
    fc = next(op for op in g.operators if opcode(m, op) == S.BuiltinOperator.FULLY_CONNECTED)
    norm = read(
        constants(m, g, next(op for op in g.operators if opcode(m, op) != S.BuiltinOperator.FULLY_CONNECTED))[0]
    )[0]
    assert np.allclose(norm, norm.flat[0]), "the soft-token norm has a learned scale: HF's embed_vision has none"
    return {"embed_vision.embedding_projection.weight": read(constants(m, g, fc)[0])[0]}


def map_text(m, g, read):
    """Text tower: INT4 products (q/k/v by the norm they feed, gate by the GELU it feeds), norms, PLE, layer scalars, head."""
    p, out, prod = "language_model", {}, producers(g)
    fc, gelu = S.BuiltinOperator.FULLY_CONNECTED, S.BuiltinOperator.GELU
    gate, gating = {}, {}  # layer -> id of the gate product; layer -> [(id, weight)] of both gating products

    def weight(op):
        return read(constants(m, g, op)[0])[0]

    for op in g.operators:
        n, code = tname(g, op.outputs[0]), opcode(m, op)
        k = layer_of(n)
        hit = re.search(r"/(query|key|value)_norm/composite\d*$", n)
        if hit and k is not None:
            src = trace_back(m, g, prod, [i for i in op.inputs if i not in constants(m, g, op)][0], fc)
            out[f"{p}.layers.{k}.self_attn.{hit.group(1)[0]}_proj.weight"] = weight(src)
        elif (
            code == gelu
            and k is not None
            and "gating_einsum" in tname(g, trace_back(m, g, prod, op.inputs[0], fc).outputs[0])
        ):
            # [Agent] Each text layer has two GELUs (MLP gate and PLE gate): only the one fed by a gating product names the gate.
            src = trace_back(m, g, prod, op.inputs[0], fc)
            out[f"{p}.layers.{k}.mlp.gate_proj.weight"], gate[k] = weight(src), id(src)
        elif code == fc and k is not None:
            names = {
                "attn_vec_einsum": "self_attn.o_proj",
                "mlp/linear": "mlp.down_proj",
                "gating_einsum": "mlp.up_or_gate",
                "per_layer_embedding_gate": "ple_block.per_layer_input_gate",
                "per_layer_embedding_projection": "ple_block.per_layer_projection",
            }
            hitn = next((v for key, v in names.items() if key in n), None)
            if hitn == "mlp.up_or_gate":
                gating.setdefault(k, []).append((id(op), weight(op)))
            elif hitn:
                out[f"{p}.layers.{k}.{hitn}.weight"] = weight(op)
        elif code == S.BuiltinOperator.MUL and k is not None and "skip_scale" in n and constants(m, g, op):
            out[f"{p}.layers.{k}.layer_scalar"] = weight(op).reshape(1)
        elif code == fc and "per_layer_model_projection" in n:
            # [Agent] Google's PLE projection is HF's / 512 (a fold; an RMS norm follows, so only eps sees it): undone here.
            out[f"{p}.ple.per_layer_model_projection.weight"] = 512.0 * weight(op)
        elif "per_layer_embedding_projection_norm/composite" in n:
            out[f"{p}.ple.per_layer_projection_norm.weight"] = weight(op)
        elif "final_norm/composite" in n:
            out[f"{p}.norm.weight"] = weight(op)
        elif code == fc and n.startswith("StatefulPartitionedCall"):
            out[f"{p}.embedding_projection.weight"] = weight(op)
    for k, products in gating.items():
        out[f"{p}.layers.{k}.mlp.up_proj.weight"] = next(w for i, w in products if i != gate[k])
    return out | norm_scales(m, g, read, p)


def map_specials(raw, secs):
    """The 4 special token rows in HF's table convention (stored raw; the model multiplies them by sqrt(512) at lookup)."""
    m, g, read = load_graph(raw, secs, CFG["embedder"])
    table = next(
        read(constants(m, g, op)[0])[0] for op in g.operators if opcode(m, op) == S.BuiltinOperator.EMBEDDING_LOOKUP
    )
    mul = next(float(read(constants(m, g, op)[0])[0]) for op in g.operators if opcode(m, op) == S.BuiltinOperator.MUL)
    m2, g2, read2 = load_graph(raw, secs, CFG["eoi"])
    rows = table[list(CFG["special"])].copy()
    rows[CFG["special"].index(258882)] = read2(g2.outputs[0])[0].reshape(-1) / mul
    return table, rows, mul


def convert(bundle, out):
    raw = open(bundle, "rb").read()
    secs = read_sections(raw)
    sd = map_vision(*load_graph(raw, secs, CFG["vision"]))
    sd |= map_adapter(*load_graph(raw, secs, CFG["adapter"]))
    sd |= map_text(*load_graph(raw, secs, CFG["text"]))
    table, rows, mul = map_specials(raw, secs)
    table[list(CFG["special"])] = rows
    sd["language_model.embed_tokens.weight"] = table
    sd["_google.embed_scale"] = np.array([mul], np.float32)
    tensors = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}
    save_file(tensors, out, metadata={"format": "pt", "source": bundle.rsplit("/", 1)[-1]})
    print(len(tensors), "tensors ->", out)
