"""Plan 05 step 2: builds the weight variants (D1) from the fp32 exports, with tensor-aligned data shards (D4).

  .venv313/Scripts/python.exe tools/webgpu/convert_precision.py                 # every variant
  .venv313/Scripts/python.exe tools/webgpu/convert_precision.py --variant w16   # repeatable

Reads build/webgpu/dit/base/{text,step_mha}.onnx and build/webgpu/vocos/vocos.onnx. Writes
build/webgpu/variants/<variant>/:
  text.onnx, step.onnx, vocos.onnx   the graphs, external data in <graph>.data.<k> at fp32-layout offsets
  <graph>.data.<k>                   the stored shards, at most 64 MiB each, split on tensor boundaries
  variant.json                       per shard, every tensor's stored dtype and both offsets (what the loader
                                     needs to widen a shard into the layout ORT reads), and the checks' results
  approx_weights.safetensors         the approximated checkpoint tensors, by state-dict name ("dit:..." and
                                     "vocos:..."), for golden data made with the same weights (D2)
  layout/                            the shards widened, as ORT reads them; for ORT CPU checks, not published

Stored dtypes: f32 and raw (as the layout), f16 (widened to fp32), i8 (int8 in blocks of 32 values, one fp32
scale per block, symmetric; widened to q * scale in float32). Only checkpoint tensors are approximated; they
are found by value (directly or transposed) in the checkpoints. Everything else, the precomputed tables
included, stays as exported. Variants:
  fp32   the exports, sharded
  w16    checkpoint tensors stored as f16, fp32 graphs
  w8     DiT checkpoint tensors stored as i8, the vocoder's as f16, fp32 graphs
  fp16   DiT graphs converted to fp16 compute, keeping fp32 for the time embedding, the table readers and the
         CFG/ODE tail; the vocoder fp32 with f16 storage
  q8/q4  DiT Gemm nodes rewritten as MatMul + Add, then MatMulNBits (8-bit block 128, 4-bit block 32,
         asymmetric, accuracy_level unset, fp32 activations); other checkpoint tensors f16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import onnxruntime as ort  # pyright: ignore[reportMissingImports]  # no Python 3.15 wheel; run in .venv313
import torch
from huggingface_hub import hf_hub_download
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import load_external_data_for_tensor
from onnxconverter_common.float16 import convert_float_to_float16
from onnxruntime.quantization.matmul_nbits_quantizer import (  # pyright: ignore[reportMissingImports]  # no Python 3.15 wheel; run in .venv313
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from onnxruntime.quantization.onnx_model import (  # pyright: ignore[reportMissingImports]  # no Python 3.15 wheel; run in .venv313
    ONNXModel,
)
from safetensors.numpy import save_file
from safetensors.torch import load_file


sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity import GoldenCase  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
SOURCES = {
    "text": REPO / "build/webgpu/dit/base/text.onnx",
    "step": REPO / "build/webgpu/dit/base/step_mha.onnx",
    "vocos": REPO / "build/webgpu/vocos/vocos.onnx",
}
OUT = REPO / "build/webgpu/variants"
VARIANTS = ["fp32", "w16", "w8", "fp16", "q8", "q4"]
QUANT = {"q8": (8, 128), "q4": (4, 32)}
SHARD_BYTES = 64 << 20
ALIGN = 4096  # as the exporter aligns large tensors
INLINE_BYTES = 1024  # smaller initializers stay in the graph, unapproximated
I8_BLOCK = 32
IR_MAX = 13  # ORT 1.31.1 refuses newer
CHECK_CASE = "en-short"
TABLES = {"text": ["te.freqs_cis"], "step": ["rope_tables", "unsqueeze_1"], "vocos": []}  # precomputed by the export


@dataclass
class Match:
    owner: str  # "dit" or "vocos"
    name: str  # state-dict name
    transposed: bool  # the graph holds the checkpoint tensor transposed


# ---- checkpoints and matching ----


def checkpoints() -> dict[str, dict[str, np.ndarray]]:
    dit_path = hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors")
    dit = {
        k.removeprefix("ema_model."): v.float().numpy()
        for k, v in load_file(dit_path).items()
        if k.startswith("ema_model.") and v.is_floating_point()
    }
    vocos_path = hf_hub_download("charactr/vocos-mel-24khz", "pytorch_model.bin")
    state = torch.load(vocos_path, map_location="cpu", weights_only=True)
    vocos = {k: v.float().numpy() for k, v in state.items() if v.is_floating_point()}
    return {"dit": dit, "vocos": vocos}


def digest(a: np.ndarray) -> tuple[tuple[int, ...], bytes]:
    return a.shape, hashlib.blake2b(np.ascontiguousarray(a, np.float32).tobytes(), digest_size=16).digest()


def index(ckpts: dict[str, dict[str, np.ndarray]]) -> dict[tuple[tuple[int, ...], bytes], Match]:
    out: dict[tuple[tuple[int, ...], bytes], Match] = {}
    for owner, tensors in ckpts.items():
        for name, a in tensors.items():
            out.setdefault(digest(a), Match(owner, name, False))
            if a.ndim == 2:
                out.setdefault(digest(a.T), Match(owner, name, True))
    return out


def match(model: onnx.ModelProto, idx: dict[tuple[tuple[int, ...], bytes], Match]) -> dict[str, Match]:
    found: dict[str, Match] = {}
    for init in model.graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        a = numpy_helper.to_array(init)
        if a.nbytes < INLINE_BYTES:
            continue
        m = idx.get(digest(a))
        if m is not None:
            found[init.name] = m
    return found


# ---- graph rewrites ----


def consumers(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    out: dict[str, list[onnx.NodeProto]] = {}
    for n in graph.node:
        for x in n.input:
            out.setdefault(x, []).append(n)
    return out


def ancestors(graph: onnx.GraphProto, names: Iterable[str]) -> set[str]:
    """Node names on any path into the given tensors."""
    producer = {o: n for n in graph.node for o in n.output}
    seen: set[str] = set()
    todo = [producer[x] for x in names if x in producer]
    while todo:
        n = todo.pop()
        if n.name in seen:
            continue
        seen.add(n.name)
        todo += [producer[x] for x in n.input if x in producer]
    return seen


def fp32_nodes(key: str, model: onnx.ModelProto) -> list[str]:
    """The nodes D2 keeps in fp32 when converting to fp16 compute."""
    g = model.graph
    cons = consumers(g)
    keep: set[str] = set()
    for t in TABLES[key]:  # the table readers, so the tables themselves stay fp32
        keep |= {n.name for n in cons.get(t, [])}
    if key == "step":
        # the time embedding: everything from the time input up to the time MLP's first linear
        todo = list(cons.get("time", []))
        seen: set[str] = set()  # not `keep`: a table reader on the way must still be walked through
        while todo:
            n = todo.pop()
            if n.op_type in ("Gemm", "MatMul") or n.name in seen:
                continue
            seen.add(n.name)
            keep.add(n.name)
            todo += [c for o in n.output for c in cons.get(o, [])]
        # the CFG combination and the Euler update: what v and x_next need beyond pred
        keep |= ancestors(g, ["v", "x_next"]) - ancestors(g, ["pred"])
    return sorted(keep)


def to_fp16(key: str, model: onnx.ModelProto) -> onnx.ModelProto:
    # 2^-24 and 65504 instead of the converter's 1e-7 and 1e4: closest to plain rounding of the weights
    blocked = fp32_nodes(key, model)
    io_types = {o.name: o.type.tensor_type.elem_type for o in model.graph.output}
    tables = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer if i.name in TABLES[key]}
    converted = convert_float_to_float16(
        model, min_positive_val=2**-24, max_finite_val=65504.0, keep_io_types=True, node_block_list=blocked,
    )  # fmt: skip
    keep_fp32_between_blocked(converted, set(blocked))
    restore_tables(converted, tables)
    for o in converted.graph.output:  # keep_io_types, which the converter breaks for blocked producers
        o.type.tensor_type.elem_type = io_types[o.name]
    return converted


def restore_tables(model: onnx.ModelProto, tables: dict[str, np.ndarray]) -> None:
    """The converter rounds every float initializer, blocked readers or not, and casts it back for them. The
    tables get their fp32 values back, and their readers (all blocked, fp32_nodes) read them directly."""
    g = model.graph
    for init in g.initializer:
        if init.name in tables:
            init.CopyFrom(numpy_helper.from_array(tables[init.name], init.name))
    for cast_node in [n for n in g.node if n.op_type == "Cast" and n.input[0] in tables]:
        to = {a.name: helper.get_attribute_value(a) for a in cast_node.attribute}["to"]
        if to != TensorProto.FLOAT:
            raise RuntimeError(f"{cast_node.input[0]}: read in fp16 by {cast_node.name}")
        for n in g.node:
            n.input[:] = [cast_node.input[0] if x == cast_node.output[0] else x for x in n.input]
        g.node.remove(cast_node)
    readers = {n.name for n in g.node for x in n.input if x in tables}
    assert readers, "no table readers left"


def keep_fp32_between_blocked(model: onnx.ModelProto, blocked: set[str]) -> None:
    """The converter casts every blocked node's output to fp16, even where only blocked nodes, casts back to
    fp32 or graph outputs read it. That rounds the time embedding's argument before Sin and Cos, and makes v
    and x_next fp16. Here those readers get the fp32 value, and the fp16 cast stays only for fp16 readers."""
    g = model.graph
    nodes = {n.name: n for n in g.node}
    producer = {o: n for n in g.node for o in n.output}
    outputs = {o.name for o in g.output}

    def fp32_reader(n: onnx.NodeProto) -> bool:
        to = {a.name: helper.get_attribute_value(a) for a in n.attribute}.get("to")
        return n.name in blocked or (n.op_type == "Cast" and to == TensorProto.FLOAT)

    for cast_node in [n for n in g.node if n.op_type == "Cast" and n.name.endswith("_output_cast0")]:
        to = {a.name: helper.get_attribute_value(a) for a in cast_node.attribute}["to"]
        src = producer.get(cast_node.input[0])
        if to != TensorProto.FLOAT16 or src is None or src.name not in blocked:
            continue
        fp32, fp16 = cast_node.input[0], cast_node.output[0]
        readers = [n for n in g.node if fp16 in n.input]
        for n in readers:
            if fp32_reader(n):
                n.input[:] = [fp32 if x == fp16 else x for x in n.input]
        if fp16 in outputs:  # the graph output takes the fp32 value under its own name
            src.output[:] = [fp16 if x == fp32 else x for x in src.output]
            for n in g.node:
                n.input[:] = [fp16 if x == fp32 else x for x in n.input]
            if any(not fp32_reader(n) for n in readers):
                raise RuntimeError(f"{fp16}: a graph output that fp16 nodes also read")
            g.node.remove(nodes[cast_node.name])
        elif all(fp32_reader(n) for n in readers):
            g.node.remove(nodes[cast_node.name])
    del g.value_info[:]  # stale dtypes; ORT infers them again


def gemm_to_matmul(model: onnx.ModelProto) -> int:
    """Gemm(A, W, C, transB=1) with constant W becomes MatMul(A, W^T) + C, so the quantizer sees it."""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    nodes: list[onnx.NodeProto] = []
    rewritten = 0
    for n in g.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        ok = (
            n.op_type == "Gemm" and n.input[1] in inits and attrs.get("transA", 0) == 0
            and attrs.get("alpha", 1.0) == 1.0 and attrs.get("beta", 1.0) == 1.0
        )  # fmt: skip
        if not ok:
            nodes.append(n)
            continue
        w = numpy_helper.to_array(inits[n.input[1]])
        wt = np.ascontiguousarray(w.T if attrs.get("transB", 0) else w)
        name = f"{n.input[1]}_kn"
        g.initializer.append(numpy_helper.from_array(wt, name))
        has_bias = len(n.input) > 2 and n.input[2]
        mm_out = f"{n.output[0]}_mm" if has_bias else n.output[0]
        nodes.append(helper.make_node("MatMul", [n.input[0], name], [mm_out], name=f"{n.name}_matmul"))
        if has_bias:
            nodes.append(helper.make_node("Add", [mm_out, n.input[2]], [n.output[0]], name=f"{n.name}_add"))
        rewritten += 1
    del g.node[:]
    g.node.extend(nodes)
    used = {x for n in g.node for x in n.input}
    keep = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    return rewritten


def quantize(model: onnx.ModelProto, bits: int, block: int) -> tuple[onnx.ModelProto, dict[str, str]]:
    """MatMulNBits on every MatMul with a constant B. Returns the model and, per MatMulNBits node, the name
    of the fp32 initializer it replaced (by the shared output name)."""
    inits = {i.name for i in model.graph.initializer}
    b_of = {n.output[0]: n.input[1] for n in model.graph.node if n.op_type == "MatMul" and n.input[1] in inits}
    q = MatMulNBitsQuantizer(
        model, algo_config=DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=False, bits=bits)
    )
    q.process()
    qm = cast(ONNXModel, q.model).model  # an ONNXModel at runtime
    replaced = {n.name: b_of[n.output[0]] for n in qm.graph.node if n.op_type == "MatMulNBits"}
    return qm, replaced


def dequantize_node(model: onnx.ModelProto, node: onnx.NodeProto) -> np.ndarray:
    """A MatMulNBits node's weight as (K, N) float32, from what it stores (probe_kernels.py checked the
    layout against ORT CPU and WebGPU)."""
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    k, n, bits, block = attrs["K"], attrs["N"], attrs["bits"], attrs["block_size"]
    inits = {i.name: i for i in model.graph.initializer}
    packed = numpy_helper.to_array(inits[node.input[1]])
    scales = numpy_helper.to_array(inits[node.input[2]]).astype(np.float32)
    zp = numpy_helper.to_array(inits[node.input[3]]) if len(node.input) > 3 and node.input[3] else None
    blocks = (k + block - 1) // block
    raw = packed.reshape(n, -1)
    q = np.stack([raw & 0x0F, raw >> 4], -1).reshape(n, -1) if bits == 4 else raw.astype(np.int64)
    q = q[:, : blocks * block].reshape(n, blocks, block).astype(np.float32)
    if zp is None:
        z = np.full((n, blocks), 2 ** (bits - 1), np.float32)
    elif bits == 4:
        zr = zp.reshape(n, -1)
        z = np.stack([zr & 0x0F, zr >> 4], -1).reshape(n, -1)[:, :blocks].astype(np.float32)
    else:
        z = zp.reshape(n, blocks).astype(np.float32)
    deq = (q - z[..., None]) * scales.reshape(n, blocks)[..., None]  # float32, as the kernel
    return np.ascontiguousarray(deq.reshape(n, blocks * block)[:, :k].T)


# ---- storage encodings ----


def encode(a: np.ndarray, dtype: str) -> tuple[bytes, np.ndarray, dict[str, int]]:
    """(stored bytes, the layout array ORT reads, extra fields for the table)."""
    if dtype == "f16":
        h = a.astype(np.float16)
        return h.tobytes(), h.astype(np.float32), {}
    if dtype == "i8":
        flat = a.astype(np.float32).ravel()
        blocks = (flat.size + I8_BLOCK - 1) // I8_BLOCK
        padded = np.zeros(blocks * I8_BLOCK, np.float32)
        padded[: flat.size] = flat
        b = padded.reshape(blocks, I8_BLOCK)
        scale = (np.abs(b).max(1) / np.float32(127)).astype(np.float32)
        safe = np.where(scale > 0, scale, np.float32(1))
        q = np.clip(np.rint(b / safe[:, None]), -127, 127).astype(np.int8)
        layout = (q.astype(np.float32) * scale[:, None]).astype(np.float32).ravel()[: flat.size].reshape(a.shape)
        qb = q.ravel()[: flat.size].tobytes()
        pad = (-len(qb)) % 4
        return qb + bytes(pad) + scale.tobytes(), layout, {"scalesOffset": len(qb) + pad, "block": I8_BLOCK}
    return a.tobytes(), a, {}


def widen(stored: bytes, t: dict[str, Any]) -> bytes:  # Any: a table entry
    """What the browser loader does: one tensor's stored bytes to its fp32 layout bytes."""
    if t["dtype"] == "f16":
        return np.frombuffer(stored, np.float16).astype(np.float32).tobytes()
    if t["dtype"] == "i8":
        count, block = t["count"], t["block"]
        q = np.frombuffer(stored, np.int8, count=count).astype(np.float32)
        scale = np.frombuffer(stored, np.float32, offset=t["scalesOffset"])
        return (q * np.repeat(scale, block)[:count]).astype(np.float32).tobytes()
    return stored


def save(model: onnx.ModelProto, out: Path, key: str, dtypes: dict[str, str]) -> dict[str, Any]:  # Any: json
    """Writes <key>.onnx with tensor-aligned external data and the stored shards. If any tensor is stored in
    another dtype than ORT reads, the graph and every shard also go to layout/, widened (copies: onnx's
    checker refuses hard-linked data)."""
    shards: list[dict[str, Any]] = []  # Any: json
    stored_files: list[bytearray] = []
    layout_files: list[bytearray] = []
    # a graph with nothing to widen stores exactly its layout, so its shards use the layout's alignment
    widened = any(dtypes.get(i.name) in ("f16", "i8") for i in model.graph.initializer)
    stored_align = 64 if widened else ALIGN
    for init in model.graph.initializer:
        a = numpy_helper.to_array(init)
        if a.nbytes < INLINE_BYTES:
            continue
        dtype = dtypes.get(init.name, "f32" if init.data_type == TensorProto.FLOAT else "raw")
        stored, layout_arr, extra = encode(a, dtype)
        layout = np.ascontiguousarray(layout_arr).tobytes()
        if not shards or len(layout_files[-1]) + (-len(layout_files[-1]) % ALIGN) + len(layout) > SHARD_BYTES:
            shards.append({"file": f"{key}.data.{len(shards)}", "tensors": []})
            stored_files.append(bytearray())
            layout_files.append(bytearray())
        lf, sf = layout_files[-1], stored_files[-1]
        lf.extend(bytes(-len(lf) % ALIGN))
        sf.extend(bytes(-len(sf) % stored_align))
        entry = {
            "name": init.name, "dtype": dtype, "count": int(a.size), "shape": list(a.shape),
            "layoutOffset": len(lf), "layoutLength": len(layout),
            "storedOffset": len(sf), "storedLength": len(stored), **extra,
        }  # fmt: skip
        shards[-1]["tensors"].append(entry)
        lf.extend(layout)
        sf.extend(stored)
        init.ClearField("raw_data")
        for field in ("float_data", "int32_data", "int64_data", "double_data", "uint64_data", "string_data"):
            init.ClearField(field)
        init.data_location = TensorProto.EXTERNAL
        del init.external_data[:]
        for k, v in (("location", shards[-1]["file"]), ("offset", entry["layoutOffset"]), ("length", len(layout))):
            init.external_data.append(onnx.StringStringEntryProto(key=k, value=str(v)))
    layout_dir = out / "layout" if widened else out
    layout_dir.mkdir(parents=True, exist_ok=True)
    for shard, lf, sf in zip(shards, layout_files, stored_files, strict=True):
        (out / shard["file"]).write_bytes(sf)
        if widened:
            (layout_dir / shard["file"]).write_bytes(lf)
        shard["layoutSize"], shard["storedSize"] = len(lf), len(sf)
    onnx.save(model, str(out / f"{key}.onnx"))
    if widened:
        shutil.copyfile(out / f"{key}.onnx", layout_dir / f"{key}.onnx")
    return {
        "graph": f"{key}.onnx", "graphSize": (out / f"{key}.onnx").stat().st_size, "shards": shards,
        "layoutDir": "layout" if widened else ".",
    }  # fmt: skip


# ---- one variant ----


def build(variant: str, idx: dict[tuple[tuple[int, ...], bytes], Match]) -> dict[str, Any]:  # Any: json
    out = OUT / variant
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    graphs: dict[str, Any] = {}  # Any: json
    approx: dict[str, np.ndarray] = {}
    notes: dict[str, Any] = {}  # Any: json
    for key, src in SOURCES.items():
        model = onnx.load(str(src), load_external_data=True)
        dit = key != "vocos"
        replaced: dict[str, str] = {}
        if variant == "fp16" and dit:
            notes[f"{key}.fp32Nodes"] = len(fp32_nodes(key, model))
            model = to_fp16(key, model)
        elif variant in QUANT and dit:
            notes[f"{key}.gemmRewritten"] = gemm_to_matmul(model)
            matches = match(model, idx)  # before quantizing, so the replaced B tensors are matched
            model, replaced = quantize(model, *QUANT[variant])
            for node in model.graph.node:
                if node.op_type == "MatMulNBits":
                    m = matches.get(replaced[node.name])
                    if m is None:
                        raise RuntimeError(f"{key}: {node.name} quantizes a tensor not in the checkpoint")
                    w = dequantize_node(model, node)  # (K, N), as the graph held B
                    approx[f"{m.owner}:{m.name}"] = np.ascontiguousarray(w.T if m.transposed else w)
        matches = match(model, idx)
        stored = "f32" if variant == "fp32" else "i8" if variant == "w8" and dit else "f16"
        dtypes = {name: stored for name in matches} if variant != "fp16" or not dit else {}
        graphs[key] = save(model, out, key, dtypes)
        if variant not in ("fp32", "fp16") or not dit:
            for entry in (t for s in graphs[key]["shards"] for t in s["tensors"]):
                m = matches.get(entry["name"])
                if m is not None and entry["dtype"] in ("f16", "i8"):
                    layout = read_layout(out, graphs[key], entry["name"])
                    approx[f"{m.owner}:{m.name}"] = np.ascontiguousarray(layout.T if m.transposed else layout)
        notes[f"{key}.checkpointTensors"] = len(matches) + len(replaced)
        notes[f"{key}.unmatchedFloats"] = [  # kept as exported: the precomputed tables, and anything folded
            [i.name, int(np.prod(i.dims))] for i in model.graph.initializer
            if i.data_type == TensorProto.FLOAT and i.name not in matches and int(np.prod(i.dims)) * 4 >= INLINE_BYTES
        ]  # fmt: skip
        notes[f"{key}.irVersion"] = model.ir_version
        ops: dict[str, int] = {}
        for n in model.graph.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
        notes[f"{key}.ops"] = ops
    if approx and variant != "fp16":
        save_file(approx, str(out / "approx_weights.safetensors"))
    return {"variant": variant, "graphs": graphs, "notes": notes}


def read_layout(out: Path, graph: dict[str, Any], name: str) -> np.ndarray:  # Any: json
    for s in graph["shards"]:
        for t in s["tensors"]:
            if t["name"] == name:
                with open(out / graph["layoutDir"] / s["file"], "rb") as f:
                    f.seek(t["layoutOffset"])
                    return np.frombuffer(f.read(t["layoutLength"]), np.float32).reshape(t["shape"]).copy()
    raise KeyError(name)


# ---- checks ----


def check(
    variant: str,
    summary: dict[str, Any],  # Any: json
    baseline: dict[str, list[np.ndarray]] | None,
) -> dict[str, Any]:  # Any: json
    out = OUT / variant
    res: dict[str, Any] = {}  # Any: json
    for key, graph in summary["graphs"].items():
        onnx.checker.check_model(str(out / graph["layoutDir"] / graph["graph"]))
        ir = summary["notes"][f"{key}.irVersion"]
        assert ir <= IR_MAX, f"{variant}/{key}: IR {ir}"
        for s in graph["shards"]:  # the loader's widening gives the layout bit for bit
            stored = (out / s["file"]).read_bytes()
            layout = (out / graph["layoutDir"] / s["file"]).read_bytes()
            for t in s["tensors"]:
                got = widen(stored[t["storedOffset"] : t["storedOffset"] + t["storedLength"]], t)
                want = layout[t["layoutOffset"] : t["layoutOffset"] + t["layoutLength"]]
                assert got == want, f"{variant}/{key}: {t['name']} widens differently"
    for key, graph in summary["graphs"].items():  # the precomputed tables, bit for bit as exported
        got, want = tables(out / graph["layoutDir"], key), tables(OUT / "fp32", key)
        assert got.keys() == want.keys() == set(TABLES[key]), f"{variant}/{key}: tables {sorted(got)}"
        for name in got:
            assert got[name].dtype == np.float32 and got[name].tobytes() == want[name].tobytes(), f"{variant}: {name}"
    res["outputs"] = run_cpu({k: out / g["layoutDir"] for k, g in summary["graphs"].items()})
    finite = all(bool(np.isfinite(a).all()) for v in res["outputs"].values() for a in v)
    res["finite"] = finite
    if baseline is not None:
        res["relL2VsFp32"] = {
            k: [float(np.linalg.norm(a - b) / np.linalg.norm(b)) for a, b in zip(v, baseline[k], strict=True)]
            for k, v in res["outputs"].items()
        }
    return res


def tables(folder: Path, key: str) -> dict[str, np.ndarray]:
    m = onnx.load(str(folder / f"{key}.onnx"), load_external_data=False)
    out: dict[str, np.ndarray] = {}
    for init in m.graph.initializer:
        if init.name in TABLES[key]:
            if init.data_location == TensorProto.EXTERNAL:
                load_external_data_for_tensor(init, str(folder))
            out[init.name] = numpy_helper.to_array(init)
    return out


def run_cpu(folders: dict[str, Path]) -> dict[str, list[np.ndarray]]:
    """Chunk 0, step 0 of the check case through all three graphs on ORT CPU."""
    g = GoldenCase(CHECK_CASE)
    n = g.manifest["chunks"][0]["duration"]
    opts = ort.SessionOptions()
    opts.add_session_config_entry("session.disable_prepacking", "1")
    sess = {
        k: ort.InferenceSession(str(folders[k] / f"{k}.onnx"), opts, providers=["CPUExecutionProvider"])
        for k in SOURCES
    }
    ids = np.full((1, n), -1, np.int32)
    raw = g.tensor("c0.dit.text_ids")
    ids[:, : min(n, raw.shape[1])] = raw[:, :n]
    tc, tu = cast(list[np.ndarray], sess["text"].run(None, {"text": ids}))
    grid = g.tensor("c0.plan.t")
    feeds = {
        "x": g.tensor("c0.dit.s00.x"), "step_cond": g.tensor("c0.dit.step_cond"), "text_cond": tc,
        "text_uncond": tu, "time": g.tensor("c0.dit.s00.t"),
        "cfg_strength": np.array([g.manifest["settings"]["cfg_strength"]], np.float32),
        "dt": np.array([grid[1] - grid[0]], np.float32),
    }  # fmt: skip
    step = cast(list[np.ndarray], sess["step"].run(None, feeds))
    (head,) = cast(list[np.ndarray], sess["vocos"].run(None, {"mel": g.tensor("c0.mel_out")}))
    return {"text": [tc, tu], "step": step, "vocos": [head]}


def check_gemm_rewrite(
    idx: dict[tuple[tuple[int, ...], bytes], Match], baseline: dict[str, list[np.ndarray]]
) -> dict[str, float]:
    """The Gemm rewrite alone, fp32, against the fp32 outputs: it must change nothing beyond rounding."""
    tmp = OUT / "_gemm"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for key, src in SOURCES.items():
        model = onnx.load(str(src), load_external_data=True)
        if key != "vocos":
            gemm_to_matmul(model)
        save(model, tmp, key, {})
    outputs = run_cpu(dict.fromkeys(SOURCES, tmp))
    shutil.rmtree(tmp)
    worst = {
        k: max(float(np.linalg.norm(a - b) / np.linalg.norm(b)) for a, b in zip(v, baseline[k], strict=True))
        for k, v in outputs.items()
    }
    assert max(worst.values()) < 1e-5, f"the Gemm rewrite changed the outputs: {worst}"
    return worst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", action="append", choices=VARIANTS, default=[])
    args = p.parse_args()
    wanted = args.variant or VARIANTS
    t0 = time.perf_counter()
    idx = index(checkpoints())
    print(f"checkpoints indexed in {time.perf_counter() - t0:.0f} s", flush=True)
    baseline: dict[str, list[np.ndarray]] | None = None
    if "fp32" not in wanted:
        if (OUT / "fp32/step.onnx").exists():
            baseline = run_cpu(dict.fromkeys(SOURCES, OUT / "fp32"))
        else:
            wanted = ["fp32", *wanted]  # the baseline the checks compare with
    versions = {p: version(p) for p in ("onnx", "onnxruntime", "onnxconverter-common", "torch")}
    for variant in sorted(wanted, key=VARIANTS.index):
        t0 = time.perf_counter()
        summary = build(variant, idx)
        if variant in QUANT and baseline is not None and "gemmRewriteCheck" not in summary["notes"]:
            summary["notes"]["gemmRewriteCheck"] = check_gemm_rewrite(idx, baseline)
        res = check(variant, summary, baseline)
        if variant == "fp32":
            baseline = res["outputs"]
        summary["checks"] = {k: v for k, v in res.items() if k != "outputs"}
        summary["versions"] = versions
        stored = sum(s["storedSize"] for g in summary["graphs"].values() for s in g["shards"])
        graphs = sum(g["graphSize"] for g in summary["graphs"].values())
        summary["storedBytes"], summary["graphBytes"] = stored, graphs
        (OUT / variant / "variant.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(
            f"{variant}: {(stored + graphs) / 1e9:.3f} GB stored, {sum(len(g['shards']) for g in summary['graphs'].values())}"
            f" shards, finite {res['finite']}, vs fp32 {json.dumps(summary['checks'].get('relL2VsFp32'))}"
            f" ({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
