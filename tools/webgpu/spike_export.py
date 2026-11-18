"""Plan 00 step 3: export the spike models for the ORT Web runtime decision (D3).

Exports, per sequence length n in --lengths (static shapes):
  block_n{n}     DiT block 0 of F5TTS_v1_Base. Inputs x (2, n, 1024), t (2, 1024), freqs (1, n, 64).
  matmul_n{n}    x (2n, 1024) @ W (1024, 2048), W = block 0's feed-forward up-projection.
  convpos_n{n}   InputEmbedding.conv_pos_embed, x (2, n, 1024).
  convnext_n{n}  TextEmbedding.text_blocks[0] (a ConvNeXtV2Block), x (2, n, 512).
Each model also gets an fp16 copy (`*_fp16.onnx`, fp32 inputs and outputs).

Inputs are real activations: the full DiT runs one CFG step (t = 0, noise x, the mel of
basic_ref_en.wav as cond, random token ids) and hooks record what reaches each module.
Reference outputs are PyTorch fp32 on the CPU. Everything goes to --out with a manifest.json.

Run in .venv313 (needs onnxruntime for the Python-side check):
  .venv313\\Scripts\\python.exe tools/webgpu/spike_export.py
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio.functional as AF
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from onnxconverter_common import float16
from safetensors.torch import load_file
from torch import Tensor, nn

from f5_tts.model.backbones.dit import DiT
from f5_tts.model.modules import ConvNeXtV2Block, ConvPositionEmbedding, DiTBlock, MelSpec


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "src/f5_tts/configs/F5TTS_v1_Base.yaml"
REF_WAV = REPO / "src/f5_tts/infer/examples/basic/basic_ref_en.wav"


class BlockWrap(nn.Module):
    def __init__(self, block: DiTBlock) -> None:
        super().__init__()
        self.block = block

    def forward(self, x: Tensor, t: Tensor, freqs: Tensor) -> Tensor:
        return self.block(x, t, mask=None, rope=(freqs, 1.0))


class MatMulWrap(nn.Module):
    def __init__(self, w: Tensor) -> None:
        super().__init__()
        self.w = nn.Parameter(w.contiguous(), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.matmul(x, self.w)


def load_dit(device: torch.device) -> DiT:
    cfg = OmegaConf.load(CONFIG)
    arch: dict[str, Any] = OmegaConf.to_container(cfg.model.arch)  # pyright: ignore[reportAssignmentType]  # Any: yaml values
    ckpt = hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors")
    sd = load_file(ckpt)
    prefix = "ema_model.transformer."
    sd = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    vocab_size = sd["text_embed.text_embed.weight"].shape[0] - 1
    dit = DiT(**arch, text_num_embeds=vocab_size, mel_dim=100)
    dit.load_state_dict(sd, strict=True)
    return dit.to(device=device, dtype=torch.float32).eval()


def ref_mel(device: torch.device) -> Tensor:
    wav, sr = sf.read(str(REF_WAV), dtype="float32", always_2d=True)
    audio = torch.from_numpy(wav.mean(axis=1)).unsqueeze(0)
    if sr != 24000:
        audio = AF.resample(audio, sr, 24000)
    rms = audio.square().mean().sqrt()
    if rms < 0.1:
        audio = audio * 0.1 / rms
    mel = MelSpec(mel_spec_type="vocos")(audio)  # (1, 100, frames)
    return mel.permute(0, 2, 1).to(device)  # (1, frames, 100)


def capture_inputs(dit: DiT, n: int, mel: Tensor, seed: int) -> dict[str, Tensor]:
    """Run one CFG DiT call at length n and record the inputs of the spiked modules."""
    device = mel.device
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, n, 100, generator=gen).to(device)
    n_cond = min(mel.shape[1], n // 2)
    cond = torch.zeros(1, n, 100, device=device)
    cond[:, :n_cond] = mel[:, :n_cond]
    nt = max(8, n // 4)
    text = torch.randint(0, dit.text_embed.text_embed.num_embeddings - 1, (1, nt), generator=gen).to(device)
    time = torch.zeros(1, device=device)

    rec: dict[str, list[Tensor]] = {"convpos": [], "convnext": []}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def block_hook(_m: nn.Module, args: tuple[Tensor, ...], kwargs: dict[str, Any]) -> None:  # Any: hook kwargs
        rec["x"], rec["t"] = [args[0]], [args[1]]
        rec["freqs"] = [kwargs["rope"][0]]

    def append_to(key: str) -> Callable[[nn.Module, tuple[Tensor, ...]], None]:
        def hook(_m: nn.Module, args: tuple[Tensor, ...]) -> None:
            rec[key].append(args[0])

        return hook

    hooks.append(dit.transformer_blocks[0].register_forward_pre_hook(block_hook, with_kwargs=True))
    hooks.append(dit.input_embed.conv_pos_embed.register_forward_pre_hook(append_to("convpos")))
    hooks.append(dit.text_embed.text_blocks[0].register_forward_pre_hook(append_to("convnext")))
    with torch.inference_mode():
        dit(x=x, cond=cond, text=text, time=time, cfg_infer=True, cache=False)
    for h in hooks:
        h.remove()
    dit.clear_cache()

    # conv_pos_embed and the text blocks run once per CFG half; the spike feeds both as batch 2
    return {
        "block.x": rec["x"][0],
        "block.t": rec["t"][0],
        "block.freqs": rec["freqs"][0],
        "convpos.x": torch.cat(rec["convpos"], dim=0),
        "convnext.x": torch.cat(rec["convnext"], dim=0),
    }


def export(module: nn.Module, args: tuple[Tensor, ...], names: list[str], path: Path) -> onnx.ModelProto:
    prog = torch.onnx.export(module, args, input_names=names, output_names=["out"], dynamo=True, optimize=True)
    assert prog is not None
    prog.save(str(path))
    return onnx.load(str(path))


def describe(model: onnx.ModelProto) -> dict[str, Any]:  # Any: json summary
    ops: dict[str, int] = {}
    for node in model.graph.node:
        key = f"{node.domain}::{node.op_type}" if node.domain else node.op_type
        ops[key] = ops.get(key, 0) + 1
    opset = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
    return {"opset": opset, "nodes": len(model.graph.node), "ops": dict(sorted(ops.items()))}


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-30))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 2048])
    p.add_argument("--out", type=Path, default=REPO / "web/spikes/ort-block/public/data")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dit = load_dit(device)
    mel = ref_mel(device)

    block0 = dit.transformer_blocks[0]
    assert isinstance(block0, DiTBlock)
    convpos = dit.input_embed.conv_pos_embed
    assert isinstance(convpos, ConvPositionEmbedding)
    convnext = dit.text_embed.text_blocks[0]
    assert isinstance(convnext, ConvNeXtV2Block)
    project_in = block0.ff.ff[0]
    assert isinstance(project_in, nn.Sequential)
    ff_up = project_in[0]
    assert isinstance(ff_up, nn.Linear)

    # the hand-written WGSL matmul baseline needs W itself; it is the same for every n
    np.save(out / "matmul_w.npy", ff_up.weight.detach().t().contiguous().cpu().numpy())

    cpu = torch.device("cpu")
    mods: dict[str, nn.Module] = {
        "block": BlockWrap(block0).to(cpu).eval(),
        "matmul": MatMulWrap(ff_up.weight.detach().t()).to(cpu).eval(),
        "convpos": convpos.to(cpu).eval(),
        "convnext": convnext.to(cpu).eval(),
    }

    manifest: dict[str, Any] = {  # Any: json
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "capture_device": str(device),
        "models": [],
    }
    for n in args.lengths:
        dit.to(device)
        caps = {k: v.detach().to(cpu).float().contiguous() for k, v in capture_inputs(dit, n, mel, args.seed).items()}
        cases: dict[str, tuple[dict[str, Tensor], Callable[[], Tensor]]] = {
            "block": (
                {"x": caps["block.x"], "t": caps["block.t"], "freqs": caps["block.freqs"]},
                lambda c=caps: mods["block"](c["block.x"], c["block.t"], c["block.freqs"]),
            ),
            "matmul": (
                {"x": caps["block.x"].reshape(2 * n, 1024)},
                lambda c=caps: mods["matmul"](c["block.x"].reshape(2 * n, 1024)),
            ),
            "convpos": ({"x": caps["convpos.x"]}, lambda c=caps: mods["convpos"](c["convpos.x"])),
            "convnext": ({"x": caps["convnext.x"]}, lambda c=caps: mods["convnext"](c["convnext.x"])),
        }
        dit.to(cpu)  # frees GPU memory; the spiked modules share its weights
        for kind, (inputs, ref_fn) in cases.items():
            name = f"{kind}_n{n}"
            with torch.inference_mode():
                ref = ref_fn().contiguous()
            for k, v in inputs.items():
                np.save(out / f"{name}.in.{k}.npy", v.numpy())
            np.save(out / f"{name}.ref.npy", ref.numpy())

            model = export(mods[kind], tuple(inputs.values()), list(inputs), out / f"{name}.onnx")
            m16 = float16.convert_float_to_float16(model, keep_io_types=True)
            onnx.save(m16, str(out / f"{name}_fp16.onnx"))

            feeds = {k: v.numpy() for k, v in inputs.items()}
            checks: dict[str, Any] = {}  # Any: json
            for suffix in ("", "_fp16"):
                sess = ort.InferenceSession(str(out / f"{name}{suffix}.onnx"), providers=["CPUExecutionProvider"])
                got = sess.run(None, feeds)[0]
                assert isinstance(got, np.ndarray)
                checks[f"ort_cpu{suffix}"] = {
                    "rel_l2": rel_l2(got, ref.numpy()),
                    "max_abs": float(np.abs(got - ref.numpy()).max()),
                }
            entry = {
                "name": name,
                "kind": kind,
                "n": n,
                "inputs": {k: list(v.shape) for k, v in inputs.items()},
                "output": list(ref.shape),
                "ref_abs_max": float(ref.abs().max()),
                "ref_rms": float(ref.square().mean().sqrt()),
                "flops": flops(kind, n),
                **describe(model),
                "fp16": describe(m16),
                "python_check": checks,
            }
            manifest["models"].append(entry)
            print(name, json.dumps({"opset": entry["opset"], "nodes": entry["nodes"], **checks}))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def flops(kind: str, n: int) -> float:
    """Multiply-adds times 2, matmul-like work only (enough for a throughput figure)."""
    d, ff = 1024, 2048
    if kind == "matmul":
        return 2.0 * (2 * n) * d * ff
    if kind == "block":
        proj = 2 * n * (4 * d * d + 2 * d * ff)  # q, k, v, out; ff up and down
        attn = 2 * 16 * n * n * 64 * 2  # QK^T and AV, 16 heads of 64
        return 2.0 * (proj + attn)
    if kind == "convpos":
        return 2.0 * 2 * 2 * n * d * (d // 16) * 31
    if kind == "convnext":
        return 2.0 * 2 * n * (512 * 7 + 2 * 512 * 1024)
    return math.nan


if __name__ == "__main__":
    main()
