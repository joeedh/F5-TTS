"""Plan 02 step 3: export the Vocos vocoder (backbone plus the ISTFT head's linear layer) for ORT Web.

  vocos.onnx        mel (1, 100, T)  ->  head_out (1, T, 1026)
  vocos_debug.onnx  the same, plus backbone (1, T, 512) as a first output, for bisecting

T is dynamic. head_out is the golden c*.vocoder.head_out tap: log magnitudes in the first 513 values,
phases in the rest. The exp, clip, cos/sin and inverse STFT run in TypeScript (plan 02, D2 and D3).
One file each, no external data (54 MB for the real model).

Run in .venv313:
  .venv313\\Scripts\\python.exe tools/webgpu/export_vocos.py --model tiny
  .venv313\\Scripts\\python.exe tools/webgpu/export_vocos.py --model base
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import onnx
import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.export import Dim
from vocos import Vocos

import f5_tts.infer.utils_infer as ui


sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_model import build_tiny_vocos  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
TINY_WEIGHTS = REPO / "testdata/webgpu/tiny/weights/vocos.safetensors"
MIN_T, MAX_T = 8, 16384
EXPORT_T = 128  # example length for tracing


class VocoderGraph(nn.Module):
    """vocos.decode up to the head's linear layer: backbone, then ISTFTHead.out."""

    def __init__(self, vocos: Vocos, debug: bool) -> None:
        super().__init__()
        self.backbone = vocos.backbone
        out = getattr(vocos.head, "out")  # ISTFTHead.out; the head is typed as the FourierHead base
        assert isinstance(out, nn.Linear)
        self.out = out
        self.debug = debug

    def forward(self, mel: Tensor) -> tuple[Tensor, ...]:
        h = self.backbone(mel)  # (1, T, dim)
        head_out = self.out(h)
        return (h, head_out) if self.debug else (head_out,)


def load(model: str) -> Vocos:
    if model == "tiny":
        vocos = build_tiny_vocos()
        vocos.load_state_dict(load_file(str(TINY_WEIGHTS)), strict=True)
    else:
        vocos = ui.load_vocoder("vocos", device="cpu")
    return vocos.float().eval()


def export(vocos: Vocos, path: Path, debug: bool) -> dict[str, Any]:  # Any: json summary
    graph = VocoderGraph(vocos, debug)
    mel = torch.randn(1, 100, EXPORT_T)
    outputs = ["backbone", "head_out"] if debug else ["head_out"]
    prog = torch.onnx.export(
        graph,
        (mel,),
        input_names=["mel"],
        output_names=outputs,
        dynamic_shapes={"mel": {2: Dim("t", min=MIN_T, max=MAX_T)}},
        opset_version=20,
        dynamo=True,
        optimize=True,
    )
    assert prog is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    prog.save(str(path), external_data=False)
    model = onnx.load(str(path))
    t = model.graph.input[0].type.tensor_type.shape.dim[2]
    assert t.dim_param, f"{path.name}: T is fixed at {t.dim_value}"
    ops: dict[str, int] = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    # the backbone's exact GELU: one Gelu op per ConvNeXt block, or an Erf decomposition (plan 02, R6)
    gelu = {k: v for k, v in ops.items() if k in ("Gelu", "Erf", "Tanh")}
    return {"file": path.name, "bytes": path.stat().st_size, "nodes": len(model.graph.node), "gelu": gelu, "ops": ops}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["tiny", "base"], required=True)
    p.add_argument("--out", type=Path, help="default: testdata/webgpu/tiny/onnx or build/webgpu/vocos")
    args = p.parse_args()
    # the exporter prints emoji; a Windows console or pipe defaults to cp1252 and raises (docs/debugging.md)
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    out: Path = args.out or (
        REPO / "testdata/webgpu/tiny/onnx" if args.model == "tiny" else REPO / "build/webgpu/vocos"
    )
    vocos = load(args.model)
    with torch.no_grad():
        for debug in (False, True):
            info = export(vocos, out / ("vocos_debug.onnx" if debug else "vocos.onnx"), debug)
            print(json.dumps(info))


if __name__ == "__main__":
    main()
