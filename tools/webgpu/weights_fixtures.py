"""Plan 05 step 3: fixtures for the browser's shard widening (web/src/weights/variant.ts).

  .venv313/Scripts/python.exe tools/webgpu/weights_fixtures.py

Writes testdata/webgpu/weights/:
  f16_to_f32.bin   numpy's float32 bits of all 65536 float16 bit patterns, as uint32
  tiny/            a synthetic graph's initializers through convert_precision.py's own save(), with 16 KiB
                   shards: the stored shards, layout/ (what widening must give, byte for byte) and tiny.json
                   (the tensor table)
The tensors are each at least 1 KB, since save() keeps smaller ones in the graph. They cover every stored
dtype and the edge values: fp16 subnormals and the range ends, two int8 blocks of zeros, and a last int8 block
that is not full. One shard has only f32 and raw tensors, at stored offsets that differ from the layout's.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from onnx import helper, numpy_helper


sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_precision as cp  # noqa: E402


OUT = cp.REPO / "testdata/webgpu/weights"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bits = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32).view(np.uint32)
    bits.tofile(OUT / "f16_to_f32.bin")

    rng = np.random.default_rng(20261123)
    edge = np.concatenate(
        [
            [0.0, -0.0, 2**-24, -(2**-24), 2**-25, 3 * 2**-26, 6e-8, 6.1e-5, 65504.0, -65504.0, 65519.0],
            rng.uniform(-1e-5, 1e-5, 200),
            rng.normal(0, 1, 289),
        ]
    ).astype(np.float32)
    tensors: dict[str, tuple[np.ndarray, str]] = {
        "a_f16": (rng.normal(0, 0.05, (37, 19)).astype(np.float32), "f16"),
        "b_f16_edges": (edge, "f16"),
        "c_i8_partial": (rng.normal(0, 1, 1025).astype(np.float32), "i8"),  # a last block of one value
        "d_i8_zeros": (np.concatenate([np.zeros(64), rng.normal(0, 1, 448)]).astype(np.float32), "i8"),
        "e_i8": ((rng.normal(0, 1, (30, 100)) * 100).astype(np.float32), "i8"),
        "f_f32": (rng.normal(0, 1, 300).astype(np.float32), "f32"),
        "g_raw": (np.arange(200, dtype=np.int64), "raw"),
        "h_f16_big": (rng.normal(0, 1, 5000).astype(np.float32), "f16"),
    }
    graph = helper.make_graph(
        [], "tiny", [], [], [numpy_helper.from_array(a, name) for name, (a, _) in tensors.items()]
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 20)])
    folder = OUT / "tiny"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir()
    cp.SHARD_BYTES = 16 << 10  # several shards from small tensors
    table = cp.save(model, folder, "tiny", {name: dtype for name, (_, dtype) in tensors.items() if dtype != "raw"})
    (folder / "tiny.json").write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    (folder / "tiny.onnx").unlink()  # the tests need only the data
    (folder / "layout" / "tiny.onnx").unlink()
    print(f"{len(table['shards'])} shards -> {folder}")


if __name__ == "__main__":
    main()
