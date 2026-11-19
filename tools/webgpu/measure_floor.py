"""Noise floors and tolerances for the WebGPU port (plan 00, step 5).

Reruns each stage on CUDA in fp32 (TF32 off) with the CPU golden inputs (teacher forcing), and
compares with the CPU golden outputs. The floor of a check is the worst difference over every case,
chunk and step; its tolerance is 10 times that, rounded up to a power of ten.

  .venv313\\Scripts\\python.exe tools/webgpu/measure_floor.py

Writes testdata/webgpu/floors.json (every measurement) and testdata/webgpu/tolerances.json (the
table both parity runners read). The tiny case is measured and reported, but its random weights do not
give representative numbers (plan 00, R5), so it does not set any floor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torchaudio
from torch import Tensor, nn

import f5_tts.model.cfm as cfm_module


sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_golden import CASES, OUT_ROOT, load_models  # noqa: E402


LOG_MEL_MIN = math.log(1e-4)  # log-mel bins below this are masked (plan 00, "Tolerances")
MIN_FLOOR = 1e-7  # a measured floor of exactly 0 still gets a tolerance from fp32 resolution


def metrics(got: Tensor | np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | None]:
    a = (got.detach().float().cpu().numpy() if isinstance(got, Tensor) else got).astype(np.float64)
    b = ref.astype(np.float64)
    assert a.shape == b.shape, f"shape {a.shape} vs golden {b.shape}"
    if mask is not None:
        a, b = a[mask], b[mask]
    d = a - b
    nb = float(np.linalg.norm(b))
    small = nb < 1e-6 * b.size
    na = float(np.linalg.norm(a))
    return {
        "max_abs": float(np.abs(d).max()) if d.size else 0.0,
        "rel_l2": None if small else float(np.linalg.norm(d) / nb),
        "cos": None if small or na == 0 else float((a * b).sum() / (na * nb)),
        "ref_abs_max": float(np.abs(b).max()) if b.size else 0.0,
    }


@dataclass
class Floors:
    rows: list[dict[str, Any]] = field(default_factory=list)  # Any: json values

    def add(self, check: str, case: str, tensor: str, m: dict[str, float | None]) -> None:
        self.rows.append({"check": check, "case": case, "tensor": tensor, **m})

    def worst(self, check: str, key: str, include_tiny: bool = False) -> dict[str, Any] | None:  # Any: a row
        rows = [r for r in self.rows if r["check"] == check and r[key] is not None]
        if not include_tiny:
            rows = [r for r in rows if r["case"] != "tiny"]
        return max(rows, key=lambda r: r[key]) if rows else None


class Golden:
    def __init__(self, case: str) -> None:
        self.dir = OUT_ROOT / case
        self.manifest: dict[str, Any] = json.loads(
            (self.dir / "manifest.json").read_text(encoding="utf-8")
        )  # Any: json
        self.names = {t["name"] for t in self.manifest["tensors"]}

    def np(self, name: str) -> np.ndarray:
        return np.load(self.dir / f"{name}.npy")

    def t(self, name: str, device: torch.device) -> Tensor:
        a = self.np(name)
        return torch.from_numpy(a.astype(np.int64) if a.dtype == np.int32 else a).to(device)


def complex_stft(head_out: np.ndarray) -> np.ndarray:
    """Vocos ISTFTHead: first half log-magnitude (exp, clipped at 100), second half phase."""
    mag_log, phase = np.split(head_out.astype(np.float64), 2, axis=-1)
    mag = np.minimum(np.exp(mag_log), 100.0)
    return np.concatenate([mag * np.cos(phase), mag * np.sin(phase)], axis=-1)


def measure_case(name: str, model: Any, vocoder: Any, dev: torch.device, fl: Floors) -> None:  # Any: CFM, Vocos
    g = Golden(name)
    man = g.manifest
    nfe = man["settings"]["nfe_step"]
    dit = model.transformer

    # mel: from the golden reference audio
    with torch.inference_mode():
        mel = model.mel_spec(g.t("audio.ref", dev))
    ref_mel = g.np("mel.ref")
    fl.add("mel.log_masked", name, "mel.ref", metrics(mel, ref_mel, mask=ref_mel > LOG_MEL_MIN))
    fl.add("mel.linear", name, "mel.ref", metrics(mel.exp(), np.exp(ref_mel)))

    # audio: downmix, RMS raise and resample on CUDA (only en-prep resamples)
    if man["prep"]["sample_rate"] != 24000:
        prep = g.t("prep.audio", dev)
        mono = prep.mean(dim=0, keepdim=True)
        rms = torch.sqrt(torch.mean(torch.square(mono)))
        mono = mono * 0.1 / rms if rms < 0.1 else mono
        res = torchaudio.transforms.Resample(man["prep"]["sample_rate"], 24000).to(dev)(mono)
        fl.add("audio.resample", name, "audio.ref", metrics(res, g.np("audio.ref")))

    # dit: every call teacher-forced; internals at the steps the golden data has
    internal_steps = set(man["settings"]["dit_internal_steps"])
    captured: dict[str, list[Tensor]] = {}

    def capture(key: str) -> Callable[[nn.Module, tuple[Any, ...], Tensor], None]:  # Any: hook arguments
        def hook(_m: nn.Module, _a: tuple[Any, ...], out: Tensor) -> None:  # Any: hook arguments
            captured.setdefault(key, []).append(out)

        return hook

    hooks = [dit.time_embed.register_forward_hook(capture("time_embed"))]
    hooks.append(dit.text_embed.register_forward_hook(capture("text_embed")))
    hooks.append(dit.input_embed.register_forward_hook(capture("input_embed")))
    for i, block in enumerate(dit.transformer_blocks):
        hooks.append(block.register_forward_hook(capture(f"block{i:02d}")))
    hooks.append(dit.norm_out.register_forward_hook(capture("norm_out")))

    for c in range(len(man["chunks"])):
        step_cond = g.t(f"c{c}.dit.step_cond", dev)
        text = g.t(f"c{c}.dit.text_ids", dev)
        for s in range(nfe):
            p = f"c{c}.dit.s{s:02d}"
            captured.clear()
            with torch.inference_mode():
                out = dit(
                    x=g.t(f"{p}.x", dev),
                    cond=step_cond,
                    text=text,
                    time=g.t(f"{p}.t", dev)[0],
                    cfg_infer=True,
                    cache=False,
                )
            cond, uncond = torch.chunk(out, 2, dim=0)
            fl.add("dit.out", name, f"{p}.out_cond", metrics(cond, g.np(f"{p}.out_cond")))
            fl.add("dit.out", name, f"{p}.out_uncond", metrics(uncond, g.np(f"{p}.out_uncond")))
            if c == 0 and s in internal_steps:
                ip = f"c0.dit_internal.s{s:02d}"
                for key, outs in captured.items():
                    for k, o in enumerate(outs):
                        suffix = "_uncond" if k == 1 else ""
                        gname = f"c0.dit_internal.{key}{suffix}" if key == "text_embed" else f"{ip}.{key}{suffix}"
                        if gname in g.names and (key != "text_embed" or s == 0):
                            check = "dit_internal.block" if key.startswith("block") else f"dit_internal.{key}"
                            fl.add(check, name, gname, metrics(o, g.np(gname)))
    for h in hooks:
        h.remove()

    # vocoder: from the golden mel_out of each chunk
    head_out_rec: list[Tensor] = []
    head_hook = getattr(vocoder.head, "out").register_forward_hook(
        lambda _m, _a, o: head_out_rec.append(o)  # pyright: ignore[reportUnknownLambdaType]  # hook signature
    )
    raised, rms = man["audio"]["raised"], man["audio"]["rms"]
    for c in range(len(man["chunks"])):
        mel_out = g.t(f"c{c}.mel_out", dev)
        head_out_rec.clear()
        with torch.inference_mode():
            backbone = vocoder.backbone(mel_out)
            wav = vocoder.decode(mel_out)
        if raised:
            wav = wav * rms / 0.1
        fl.add("vocoder.backbone", name, f"c{c}.vocoder.backbone", metrics(backbone, g.np(f"c{c}.vocoder.backbone")))
        ho = head_out_rec[0].float().cpu().numpy()
        gho = g.np(f"c{c}.vocoder.head_out")
        fl.add("vocoder.head_out", name, f"c{c}.vocoder.head_out", metrics(ho, gho))
        fl.add("vocoder.stft_complex", name, f"c{c}.vocoder.head_out", metrics(complex_stft(ho), complex_stft(gho)))
        fl.add("vocoder.wav", name, f"c{c}.vocoder.wav", metrics(wav.squeeze(), g.np(f"c{c}.vocoder.wav")))
    head_hook.remove()

    # free-running ODE with the golden noise injected; also gives the time grid
    real_odeint = cfm_module.odeint
    for c, chunk in enumerate(man["chunks"]):
        y0 = g.t(f"c{c}.noise.y0", dev)
        grid: list[Tensor] = []

        def odeint(fn: Callable[..., Tensor], _y0: Tensor, t: Tensor, **kw: Any) -> Tensor:  # Any: options
            grid.append(t)
            return cast(Tensor, real_odeint(fn, y0, t, **kw))  # euler returns the trajectory tensor

        cfm_module.odeint = odeint  # pyright: ignore[reportAttributeAccessIssue]  # restored below
        try:
            with torch.inference_mode():
                out, _ = model.sample(
                    cond=g.t("audio.ref", dev),
                    text=g.t(f"c{c}.dit.text_ids", dev),
                    duration=chunk["duration"],
                    steps=nfe,
                    cfg_strength=man["settings"]["cfg_strength"],
                    sway_sampling_coef=man["settings"]["sway_sampling_coef"],
                )
        finally:
            cfm_module.odeint = real_odeint
        fl.add("plan.t", name, f"c{c}.plan.t", metrics(grid[0], g.np(f"c{c}.plan.t")))
        mel_out = out[:, chunk["ref_audio_len"] :, :].permute(0, 2, 1)
        fl.add("mel_out.free_running", name, f"c{c}.mel_out", metrics(mel_out, g.np(f"c{c}.mel_out")))


# check -> (metrics the pass check uses, how to read it)
CHECKS: dict[str, tuple[list[str], str]] = {
    "plan.t": (["max_abs"], "time grid after sway"),
    "audio.resample": (["max_abs"], "informational; plan 02 decides (a different resampler may be accepted)"),
    "mel.log_masked": (["max_abs"], "log mel, bins with golden linear magnitude above 1e-4"),
    "mel.linear": (["rel_l2"], "linear mel magnitude, exp of the log mel"),
    "dit.out": (["rel_l2", "max_abs"], "one DiT call (cond or uncond half), golden inputs"),
    "dit_internal.text_embed": (["rel_l2", "max_abs"], "TextEmbedding output"),
    "dit_internal.time_embed": (["rel_l2", "max_abs"], "TimestepEmbedding output"),
    "dit_internal.input_embed": (["rel_l2", "max_abs"], "InputEmbedding output"),
    "dit_internal.block": (["rel_l2", "max_abs"], "output of any of the 22 blocks, full model run up to it"),
    "dit_internal.norm_out": (["rel_l2", "max_abs"], "AdaLayerNorm_Final output"),
    "vocoder.backbone": (["rel_l2", "max_abs"], "Vocos backbone output from golden mel"),
    "vocoder.head_out": (["max_abs"], "informational; raw head output, phase half is unstable where magnitude is tiny"),
    "vocoder.stft_complex": (["rel_l2"], "complex STFT from the head output"),
    "vocoder.wav": (["rel_l2", "max_abs"], "chunk wav after output RMS rescale, golden mel"),
    "mel_out.free_running": (["rel_l2"], "free-running ODE with golden noise, cropped mel"),
}


def pow10_up(x: float) -> float:
    return float(10.0 ** math.ceil(math.log10(max(x, MIN_FLOOR) * 10.0) - 1e-9))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", action="append", choices=sorted(CASES), default=[])
    p.add_argument("--from-floors", action="store_true", help="rebuild tolerances.json from floors.json, no GPU")
    args = p.parse_args()

    if args.from_floors:
        fl = Floors(rows=json.loads((OUT_ROOT / "floors.json").read_text())["rows"])
        names = sorted({r["case"] for r in fl.rows})
        gpu = json.loads((OUT_ROOT / "tolerances.json").read_text())["measured_with"]["gpu"]
    else:
        names = args.case or sorted(CASES)
        fl, gpu = measure(names), torch.cuda.get_device_name()
        (OUT_ROOT / "floors.json").write_text(json.dumps({"rows": fl.rows}, indent=1))
    write_tolerances(fl, names, gpu)


def measure(names: list[str]) -> Floors:
    assert torch.cuda.is_available(), "needs CUDA (run in .venv313)"
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    fl = Floors()
    models: dict[str, tuple[Any, Any]] = {}  # Any: CFM, Vocos
    for name in names:
        case = CASES[name]
        if case.model not in models:
            model, vocoder = load_models(case, init_tiny=False)
            models[case.model] = (model.to(dev).eval(), vocoder.to(dev).eval())
        model, vocoder = models[case.model]
        n0 = len(fl.rows)
        measure_case(name, model, vocoder, dev, fl)
        print(f"{name}: {len(fl.rows) - n0} comparisons", flush=True)
    return fl


def write_tolerances(fl: Floors, names: list[str], gpu: str) -> None:
    table: dict[str, Any] = {}  # Any: json
    for check, (keys, note) in CHECKS.items():
        entry: dict[str, Any] = {"note": note}  # Any: json
        for key in keys:
            w = fl.worst(check, key)
            if w is None:
                continue
            entry[key] = pow10_up(w[key])
            entry[f"floor_{key}"] = w[key]
            entry[f"floor_{key}_at"] = f"{w['case']}/{w['tensor']}"
            wt = fl.worst(check, key, include_tiny=True)
            if wt is not None and wt["case"] == "tiny":
                entry[f"tiny_{key}"] = wt[key]
        table[check] = entry
    tolerances = {
        "method": "10x the worst CPU fp32 vs CUDA fp32 (TF32 off) difference over all real cases, rounded up to a"
        " power of ten; see docs/plans/webgpu-00-scope-and-parity.md, Tolerances",
        "measured_with": {
            "torch": torch.__version__,
            "gpu": gpu,
            "cases": [n for n in names if n != "tiny"],
        },
        "exact": [
            "text.max_chars",
            "text.chunks",
            "text.tokens",
            "dit.text_ids",
            "plan.duration",
            "plan.ref_audio_len",
            "plan.local_speed",
        ],
        "near_zero_rule": "relative L2 is not used where ||golden|| < 1e-6 * size; max_abs decides",
        "checks": table,
    }
    (OUT_ROOT / "tolerances.json").write_text(json.dumps(tolerances, indent=2) + "\n")
    for check, entry in table.items():
        vals = ", ".join(f"{k}={v:.3g}" for k, v in entry.items() if isinstance(v, float))
        print(f"{check}: {vals}")


if __name__ == "__main__":
    main()
