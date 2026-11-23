"""Plan 05 step 1: tiny ONNX graphs that show how ORT Web's WebGPU kernels compute in fp16, and MatMulNBits.

  .venv313/Scripts/python.exe tools/webgpu/probe_kernels.py

ORT's source is not reachable (plan 05), so each probe measures instead: one op, in the shapes and attributes
the DiT uses, on random inputs. fp16 probes cast fp32 graph inputs to fp16 inside the graph and cast the output
back, with the weights as fp16 initializers. The reference is float64 on exactly the values the graph sees,
and "ideal" is that reference rounded to fp16: what an op that accumulates in fp32 and rounds once returns.
MatMulNBits probes are made by onnxruntime's own quantizer; their reference dequantizes the stored q, zero
point and scale read back from the built graph. Each probe also runs on ORT CPU here, unoptimized, as a baseline.
Writes build/webgpu/probes/: <name>.onnx, <name>.in.bin (float32 inputs in order), <name>.ref.bin (float64),
<name>.ideal.bin (float32) and probes.json. web/bench/weights-probe/ runs them in Chrome.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort  # pyright: ignore[reportMissingImports]  # no Python 3.15 wheel; run in .venv313
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import (  # pyright: ignore[reportMissingImports]  # no Python 3.15 wheel; run in .venv313
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from onnxruntime.quantization.onnx_model import (  # pyright: ignore[reportMissingImports]  # no Python 3.15 wheel; run in .venv313
    ONNXModel,
)


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "build/webgpu/probes"
IR_VERSION = 10  # ORT 1.31.1 refuses IR 14, onnx 1.23.2's default
OPSETS = [helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)]
F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT


@dataclass
class Probe:
    name: str
    nodes: list[onnx.NodeProto]
    inputs: dict[str, np.ndarray]  # graph inputs, float32
    weights: dict[str, np.ndarray]  # initializers, in their stored dtype
    out_shape: tuple[int, ...]
    reference: Callable[[dict[str, np.ndarray]], np.ndarray]  # float64, from the values the graph sees
    fp16: bool
    note: str
    extra: dict[str, object] = field(default_factory=dict)


def r16(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16).astype(np.float64)


def model(p: Probe) -> onnx.ModelProto:
    """Graph inputs and the output are float32; fp16 probes cast inside the graph."""
    nodes = list(p.nodes)
    gin = [helper.make_tensor_value_info(k, F32, list(v.shape)) for k, v in p.inputs.items()]
    if p.fp16:
        nodes = [helper.make_node("Cast", [k], [f"{k}_h"], to=F16) for k in p.inputs] + nodes
        nodes.append(helper.make_node("Cast", ["y_h"], ["y"], to=F32))
    out = helper.make_tensor_value_info("y", F32, list(p.out_shape))
    inits = [numpy_helper.from_array(v, k) for k, v in p.weights.items()]
    g = helper.make_graph(nodes, p.name, gin, [out], inits)
    m = helper.make_model(g, opset_imports=OPSETS, ir_version=IR_VERSION)
    onnx.checker.check_model(m)
    return m


def inp(p_fp16: bool, name: str) -> str:
    return f"{name}_h" if p_fp16 else name


def out(p_fp16: bool) -> str:
    return "y_h" if p_fp16 else "y"


def matmul(name: str, m: int, k: int, n: int, fp16: bool, ones: bool, rng: np.random.Generator) -> Probe:
    dt = np.float16 if fp16 else np.float32
    a = np.ones((m, k), np.float32) if ones else rng.uniform(-1, 1, (m, k)).astype(np.float32)
    b = np.full((k, n), 1 + 2**-10, dt) if ones else rng.uniform(-1, 1, (k, n)).astype(dt)
    note = "all-ones A, B = 1 + 2^-10: fp32 accumulation gives exactly K * (1 + 2^-10)" if ones else "random"
    return Probe(
        name, [helper.make_node("MatMul", [inp(fp16, "a"), "b"], [out(fp16)])], {"a": a}, {"b": b}, (m, n),
        lambda g: (r16(g["a"]) if fp16 else g["a"].astype(np.float64)) @ b.astype(np.float64), fp16, note,
    )  # fmt: skip


def gemm(name: str, m: int, k: int, n: int, rng: np.random.Generator) -> Probe:
    w = rng.uniform(-1, 1, (n, k)).astype(np.float16)
    c = rng.uniform(-1, 1, (n,)).astype(np.float16)
    node = helper.make_node("Gemm", ["a_h", "w", "c"], ["y_h"], transB=1)
    a = rng.uniform(-1, 1, (m, k)).astype(np.float32)
    return Probe(
        name, [node], {"a": a}, {"w": w, "c": c}, (m, n),
        lambda g: r16(g["a"]) @ w.astype(np.float64).T + c.astype(np.float64), True, "transB = 1, as the export",
    )  # fmt: skip


def conv(name: str, ch: int, length: int, kernel: int, group: int, rng: np.random.Generator) -> Probe:
    w = rng.uniform(-1, 1, (ch, ch // group, kernel)).astype(np.float16)
    bias = rng.uniform(-1, 1, (ch,)).astype(np.float16)
    pad = kernel // 2
    node = helper.make_node("Conv", ["x_h", "w", "bias"], ["y_h"], kernel_shape=[kernel], group=group, pads=[pad, pad])
    x = rng.uniform(-1, 1, (1, ch, length)).astype(np.float32)

    def ref(g: dict[str, np.ndarray]) -> np.ndarray:
        xs = np.pad(r16(g["x"])[0], ((0, 0), (pad, pad)))
        per = ch // group
        y = np.zeros((ch, length))
        for o in range(ch):
            grp = o // per
            cols = xs[grp * per : (grp + 1) * per]
            for t in range(kernel):
                y[o] += w[o, :, t].astype(np.float64) @ cols[:, t : t + length]
        return (y + bias.astype(np.float64)[:, None])[None]

    return Probe(
        name, [node], {"x": x}, {"w": w, "bias": bias}, (1, ch, length), ref, True, f"K = {ch // group * kernel}"
    )


def mha(name: str, seq: int, hidden: int, heads: int, rng: np.random.Generator) -> Probe:
    q, k, v = (rng.normal(0, 2, (1, seq, hidden)).astype(np.float32) for _ in range(3))
    node = helper.make_node(
        "MultiHeadAttention", ["q_h", "k_h", "v_h"], ["y_h"], domain="com.microsoft", num_heads=heads
    )

    def ref(g: dict[str, np.ndarray]) -> np.ndarray:
        d = hidden // heads
        split = [r16(g[x])[0].reshape(seq, heads, d).transpose(1, 0, 2) for x in ("q", "k", "v")]
        s = split[0] @ split[1].transpose(0, 2, 1) / np.sqrt(d)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        return (p @ split[2]).transpose(1, 0, 2).reshape(1, seq, hidden)

    return Probe(name, [node], {"q": q, "k": k, "v": v}, {}, (1, seq, hidden), ref, True, "q, k, v ~ N(0, 2)")


def layernorm(name: str, rows: int, width: int, offset: float, rng: np.random.Generator) -> Probe:
    scale = rng.uniform(0.5, 1.5, (width,)).astype(np.float16)
    bias = rng.uniform(-1, 1, (width,)).astype(np.float16)
    node = helper.make_node(
        "LayerNormalization", ["x_h", "scale", "bias"], ["y_h"], axis=-1, epsilon=1e-6, stash_type=1
    )
    x = (offset + rng.normal(0, 1, (rows, width))).astype(np.float32)

    def ref(g: dict[str, np.ndarray]) -> np.ndarray:
        x = r16(g["x"])
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + 1e-6) * scale.astype(np.float64) + bias.astype(np.float64)

    return Probe(
        name, [node], {"x": x}, {"scale": scale, "bias": bias}, (rows, width), ref, True, f"mean {offset}, std 1"
    )


def unary(
    name: str, op: str, fn: Callable[[np.ndarray], np.ndarray], rng: np.random.Generator, approximate: str = ""
) -> Probe:
    x = rng.uniform(-6, 6, (64, 1024)).astype(np.float32)
    attrs = {"approximate": approximate} if approximate else {}
    node = helper.make_node(op, ["x_h"], ["y_h"], **attrs)
    return Probe(name, [node], {"x": x}, {}, x.shape, lambda g: fn(r16(g["x"])), True, "x ~ U(-6, 6)")


def gelu_none(x: np.ndarray) -> np.ndarray:
    from math import erf

    return 0.5 * x * (1 + np.vectorize(erf)(x / np.sqrt(2)))


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def nbits(name: str, m: int, k: int, n: int, bits: int, block: int, symmetric: bool, fp16: bool) -> Probe:
    """onnxruntime's own quantizer on a float32 MatMul; the reference dequantizes what it stored."""
    rng = np.random.default_rng(bits * 1000 + block + symmetric)
    w = rng.normal(0, 0.05, (k, n)).astype(np.float32)
    a = rng.uniform(-1, 1, (m, k)).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("MatMul", ["a", "b"], ["y"])], name,
        [helper.make_tensor_value_info("a", F32, [m, k])], [helper.make_tensor_value_info("y", F32, [m, n])],
        [numpy_helper.from_array(w, "b")],
    )  # fmt: skip
    q = MatMulNBitsQuantizer(
        helper.make_model(g, opset_imports=OPSETS, ir_version=IR_VERSION),
        algo_config=DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=symmetric, bits=bits),
    )
    q.process()
    qm = cast(ONNXModel, q.model).model  # an ONNXModel at runtime
    (node,) = [x for x in qm.graph.node if x.op_type == "MatMulNBits"]
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    inits = {i.name: numpy_helper.to_array(i) for i in qm.graph.initializer}
    packed, scales = inits[node.input[1]], inits[node.input[2]].astype(np.float32)
    zp_packed = inits[node.input[3]] if len(node.input) > 3 and node.input[3] else None
    deq = dequantize(packed, scales, zp_packed, k, n, bits, block)
    if fp16:
        scales = scales.astype(np.float16)
        deq = dequantize(packed, scales.astype(np.float32), zp_packed, k, n, bits, block)
    weights = {node.input[1]: packed, node.input[2]: scales}
    if zp_packed is not None:
        weights[node.input[3]] = zp_packed
    ins = [inp(fp16, "a"), *node.input[1:]]
    new = helper.make_node("MatMulNBits", ins, [out(fp16)], domain="com.microsoft", **attrs)
    return Probe(
        name, [new], {"a": a}, weights, (m, n),
        lambda g: (r16(g["a"]) if fp16 else g["a"].astype(np.float64)) @ deq.T, fp16,
        f"{bits}-bit, block {block}, {'symmetric' if symmetric else 'asymmetric'}, accuracy_level unset",
        {"attrs": {k2: v for k2, v in attrs.items() if isinstance(v, int)},
         "weight_relL2": float(np.linalg.norm(deq.T - w) / np.linalg.norm(w))},
    )  # fmt: skip


def dequantize(
    packed: np.ndarray, scales: np.ndarray, zp: np.ndarray | None, k: int, n: int, bits: int, block: int
) -> np.ndarray:
    """MatMulNBits' layout: B is [N, blocks, block * bits / 8], low bits first; returns the (N, K) float64 weight."""
    blocks = (k + block - 1) // block
    raw = packed.reshape(n, -1)
    if bits == 4:
        q = np.stack([raw & 0x0F, raw >> 4], -1).reshape(n, -1)
    else:
        q = raw.astype(np.int64)
    q = q[:, : blocks * block].reshape(n, blocks, block).astype(np.float64)
    if zp is None:
        z = np.full((n, blocks), 2 ** (bits - 1), np.float64)
    elif bits == 4:
        zr = zp.reshape(n, -1)
        z = np.stack([zr & 0x0F, zr >> 4], -1).reshape(n, -1)[:, :blocks].astype(np.float64)
    else:
        z = zp.reshape(n, blocks).astype(np.float64)
    deq = (q - z[..., None]) * scales.reshape(n, blocks).astype(np.float64)[..., None]
    return deq.reshape(n, blocks * block)[:, :k]


def metrics(y: np.ndarray, ref: np.ndarray, ideal: np.ndarray, fp16: bool) -> dict[str, float]:
    m = {"relL2": float(np.linalg.norm(y - ref) / np.linalg.norm(ref)), "maxAbs": float(np.abs(y - ref).max())}
    if fp16:
        m["exactFraction"] = float((y == ideal).mean())
    return m


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261123)
    probes = [
        matmul("matmul32_k1024", 256, 1024, 1024, False, False, rng),
        matmul("matmul16_ones_k4096", 16, 4096, 64, True, True, rng),
        matmul("matmul16_k1024", 256, 1024, 1024, True, False, rng),
        matmul("matmul16_k2048", 256, 2048, 1024, True, False, rng),
        gemm("gemm16_k1024_n6144", 64, 1024, 6144, rng),
        conv("conv16_convpos", 1024, 256, 31, 16, rng),
        conv("conv16_depthwise", 512, 256, 7, 512, rng),
        mha("mha16_s512", 512, 1024, 16, rng),
        layernorm("layernorm16_mean0", 64, 1024, 0.0, rng),
        layernorm("layernorm16_mean100", 64, 1024, 100.0, rng),
        unary("gelu16_tanh", "Gelu", gelu_tanh, rng, approximate="tanh"),
        unary("gelu16_none", "Gelu", gelu_none, rng),
        unary("mish16", "Mish", lambda x: x * np.tanh(np.log1p(np.exp(x))), rng),
        unary("sigmoid16", "Sigmoid", lambda x: 1 / (1 + np.exp(-x)), rng),
        unary("sin16", "Sin", np.sin, rng),
    ]
    for bits in (4, 8):
        for block in (32, 128):
            probes.append(nbits(f"nbits{bits}_b{block}_f32", 64, 1024, 1024, bits, block, False, False))
    probes.append(nbits("nbits4_b32_sym_f32", 64, 1024, 1024, 4, 32, True, False))
    probes.append(nbits("nbits4_b32_f16", 64, 1024, 1024, 4, 32, False, True))
    probes.append(nbits("nbits8_b32_f16", 64, 1024, 1024, 8, 32, False, True))

    index = []
    for p in probes:
        if p.fp16:  # inputs fp16 can hold exactly, so the input Cast changes nothing whether it runs or not
            p.inputs = {k: v.astype(np.float16).astype(np.float32) for k, v in p.inputs.items()}
        m = model(p)
        onnx.save(m, str(OUT / f"{p.name}.onnx"))
        ref = p.reference(p.inputs)
        ideal = ref.astype(np.float16).astype(np.float32) if p.fp16 else ref.astype(np.float32)
        np.concatenate([v.ravel() for v in p.inputs.values()]).astype(np.float32).tofile(OUT / f"{p.name}.in.bin")
        ref.astype(np.float64).tofile(OUT / f"{p.name}.ref.bin")
        ideal.tofile(OUT / f"{p.name}.ideal.bin")
        try:
            opts = ort.SessionOptions()  # optimized, ORT CPU merges the fp16 Cast pairs it inserts and runs fp32
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            sess = ort.InferenceSession(m.SerializeToString(), opts, providers=["CPUExecutionProvider"])
            (y,) = sess.run(None, p.inputs)
            cpu: dict[str, object] = dict(metrics(np.asarray(y, np.float64), ref, ideal, p.fp16))
        except Exception as e:  # an op without a CPU kernel at this dtype
            cpu = {"error": str(e).splitlines()[0][:200]}
        entry = {
            "name": p.name, "fp16": p.fp16, "note": p.note, "outShape": list(p.out_shape),
            "inputs": [{"name": k, "shape": list(v.shape)} for k, v in p.inputs.items()],
            "ortCpu": cpu, **p.extra,
        }  # fmt: skip
        index.append(entry)
        print(f"{p.name:24s} cpu {cpu}")
    (OUT / "probes.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
