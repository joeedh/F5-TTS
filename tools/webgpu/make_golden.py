"""Golden data for the WebGPU port (plan 00, step 4).

Runs the real PyTorch inference path (`preprocess_ref_audio_text`, `chunk_text`,
`infer_batch_process`) in fp32 on the CPU, with hooks and patches that record every
intermediate a port stage needs. Model code is not changed.

  python tools/webgpu/make_golden.py --case en-short      # or --all, or several --case
  python tools/webgpu/make_golden.py --case tiny --init-tiny   # make the tiny weights first

Output: testdata/webgpu/<case>/manifest.json plus one .npy per tensor. Only `tiny` is committed.
Every case runs twice and the two runs must give identical hashes (--no-recheck skips that).
Format and stage names: docs/plans/webgpu-00-scope-and-parity.md, "Golden data".
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch import Tensor, nn
from vocos import Vocos

import f5_tts.infer.utils_infer as ui
import f5_tts.model.cfm as cfm_module
from f5_tts.model import CFM
from f5_tts.model.backbones.dit import DiT
from f5_tts.model.utils import get_tokenizer, seed_everything


sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_model import load_tiny  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "testdata/webgpu"
EXAMPLES = REPO / "src/f5_tts/infer/examples/basic"
VOCAB = REPO / "src/f5_tts/infer/examples/vocab.txt"
SR = ui.target_sample_rate
HOP = ui.hop_length

EN_REF_TEXT = "Some call me nature, others call me mother nature."
EN_GEN = (
    "I don't really care what you call me. I've been a silent spectator, watching species evolve, "
    "empires rise and fall. But always remember, I am mighty and enduring."
)
# chunk_text splits this into 162, 172 and 6 bytes at the en reference's max_chars (175); the last
# chunk is under 10 bytes, so it gets the speed x0.3 rule
EN_MULTI = (
    EN_GEN + " Respect me and I'll nurture you; ignore me and you shall face the consequences. "
    "Every forest, every ocean, every mountain is my voice, and it is never silent, not at night. Go on."
)
ZH_REF_TEXT = "对，这就是我，万人敬仰的太乙真人。"
# 行 (xing/hang), 长 (chang/zhang), 重 (zhong/chong) are polyphones; Latin letters and digits mixed in
ZH_GEN = "我们银行的行长重新安排了长期计划，明天用AI和GPU再试一次，共3个版本。"


@dataclass
class Case:
    name: str
    model: str  # "tiny" | "base"
    ref_wav: str  # file in EXAMPLES, or a derived input made by make_input()
    ref_text: str
    gen_text: str
    nfe: int = 32
    cfg: float = 2.0
    sway: float = -1.0
    speed: float = 1.0
    fix_duration: float | None = None
    seed: int = 0
    expect_chunks: int | None = None
    derive: str | None = None  # "first1s" | "prep"


CASES: dict[str, Case] = {
    c.name: c
    for c in [
        Case("tiny", "tiny", "basic_ref_en.wav", "Some call me", "Hello there, world.", nfe=8, derive="first1s"),
        Case("en-short", "base", "basic_ref_en.wav", EN_REF_TEXT, EN_GEN, expect_chunks=1),
        Case("zh-short", "base", "basic_ref_zh.wav", ZH_REF_TEXT, ZH_GEN, expect_chunks=1),
        Case("en-multi", "base", "basic_ref_en.wav", EN_REF_TEXT, EN_MULTI, expect_chunks=3),
        Case("en-multi-fixdur", "base", "basic_ref_en.wav", EN_REF_TEXT, EN_MULTI, fix_duration=35.0, expect_chunks=3),
        Case(
            "en-prep",
            "base",
            "basic_ref_en.wav",
            EN_REF_TEXT + " Some call me nature.",
            "Listen closely when the wind moves through the trees.",
            derive="prep",
        ),
        Case("en-epss16", "base", "basic_ref_en.wav", EN_REF_TEXT, EN_GEN, nfe=16, expect_chunks=1),
    ]
}

# the two lines of infer_process this script repeats (risk R9 in the plan)
COPIED_LINES = [
    'max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * speed)',
    "gen_text_batches = chunk_text(gen_text, max_chars=max_chars)",
]


@dataclass
class Entry:
    tensor: np.ndarray
    stage: str
    axes: str


@dataclass
class Recorder:
    """Collects tensors and values while one case runs."""

    internal_steps: set[int]
    tensors: dict[str, Entry] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)  # Any: json values
    chunks: list[dict[str, Any]] = field(default_factory=list)  # Any: json values
    chunk: int = -1
    step: int = -1
    internal_calls: dict[str, int] = field(default_factory=dict)

    def put(self, name: str, t: Tensor | np.ndarray, stage: str, axes: str) -> None:
        a = t.detach().cpu().numpy() if isinstance(t, Tensor) else np.asarray(t)
        if a.dtype == np.int64 or a.dtype == np.bool_:
            a = a.astype(np.int32)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        assert len(axes.split()) == a.ndim, f"{name}: axes {axes!r} vs shape {a.shape}"
        assert name not in self.tensors, f"{name} recorded twice"
        self.tensors[name] = Entry(np.ascontiguousarray(a), stage, axes)

    def put_once(self, name: str, t: Tensor, stage: str, axes: str) -> None:
        """For inputs that must not change between calls within a chunk: stored once, checked after."""
        if name in self.tensors:
            assert np.array_equal(
                self.tensors[name].tensor, t.detach().cpu().numpy().astype(self.tensors[name].tensor.dtype)
            ), name
        else:
            self.put(name, t, stage, axes)

    @property
    def cur(self) -> dict[str, Any]:  # Any: json values
        return self.chunks[self.chunk]

    def internal(self) -> bool:
        return self.chunk == 0 and self.step in self.internal_steps


class SyncExecutor:
    """Stands in for ThreadPoolExecutor in utils_infer: runs chunks in order, on this thread (risk R1)."""

    def __init__(self, rec: Recorder) -> None:
        self.rec = rec

    def __enter__(self) -> SyncExecutor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future[Any]:  # Any: whatever the chunk function takes
        fut: Future[Any] = Future()  # Any: as above
        result = fn(*args)
        wave, _mel = result
        self.rec.put(f"c{self.rec.chunk}.vocoder.wav", np.asarray(wave, dtype=np.float32), "vocoder", "nw")
        fut.set_result(result)
        return fut


def git_commit() -> str:
    def run(*cmd: str) -> str:
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip()

    return run("git", "rev-parse", "HEAD") + (" (dirty)" if run("git", "status", "--porcelain") else "")


def make_input(case: Case, out: Path) -> Path:
    """The case's reference wav, copied or derived, as out/input.wav."""
    src = EXAMPLES / case.ref_wav
    dst = out / "input.wav"
    if case.derive is None:
        shutil.copyfile(src, dst)
        return dst
    wav, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = torch.from_numpy(wav.mean(axis=1))
    if case.derive == "first1s":
        sf.write(str(dst), mono[:sr].numpy(), sr, subtype="PCM_16")
    elif case.derive == "prep":
        # 44.1 kHz stereo, looped past 12 s, RMS 0.03: exercises downmix, resample, RMS raise and the 12 s clip
        x = AF.resample(mono.unsqueeze(0), sr, 44100)[0]
        x = torch.cat([x, x, x])
        stereo = torch.stack([x, 0.8 * x], dim=1)
        stereo = stereo * (0.03 / stereo.square().mean().sqrt())
        sf.write(str(dst), stereo.numpy(), 44100, subtype="PCM_16")
    else:
        raise ValueError(case.derive)
    return dst


def load_models(case: Case, init_tiny: bool) -> tuple[CFM, Vocos]:
    vocab_char_map, vocab_size = get_tokenizer(str(VOCAB), "custom")
    mel_kwargs = dict(n_fft=1024, hop_length=HOP, win_length=1024, n_mel_channels=100, target_sample_rate=SR)
    if case.model == "tiny":
        dit, vocoder = load_tiny(vocab_size, OUT_ROOT / "tiny/weights", init_tiny)
        model = CFM(
            transformer=dit, mel_spec_kwargs={**mel_kwargs, "mel_spec_type": "vocos"}, vocab_char_map=vocab_char_map
        )
        return model.float().eval(), vocoder
    cfg = OmegaConf.load(REPO / "src/f5_tts/configs/F5TTS_v1_Base.yaml")
    arch: dict[str, Any] = OmegaConf.to_container(cfg.model.arch)  # pyright: ignore[reportAssignmentType]  # Any: yaml values
    model = CFM(
        transformer=DiT(**arch, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs={**mel_kwargs, "mel_spec_type": "vocos"},
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_char_map,
    )
    ckpt = hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors")
    # load_model has no dtype argument; load_checkpoint picks fp16 on recent CUDA GPUs (risk R2)
    model = ui.load_checkpoint(model, ckpt, "cpu", dtype=torch.float32)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    vocoder = ui.load_vocoder("vocos", device="cpu")
    return model.eval(), vocoder


@contextlib.contextmanager
def recording(model: CFM, vocoder: Vocos, rec: Recorder, nfe: int) -> Iterator[None]:
    dit = model.transformer
    assert isinstance(dit, DiT)
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    real_odeint = cfm_module.odeint
    real_pinyin = ui.convert_char_to_pinyin
    real_executor = ui.ThreadPoolExecutor

    def odeint(fn: Callable[..., Tensor], y0: Tensor, t: Tensor, **kw: Any) -> Tensor:  # Any: odeint options
        rec.chunk += 1
        rec.step = -1
        c = rec.chunk
        rec.cur["duration"] = int(y0.shape[1])
        rec.put(f"c{c}.noise.y0", y0, "noise", "b n d")
        rec.put(f"c{c}.plan.t", t, "plan", "t")
        out = cast(Tensor, real_odeint(fn, y0, t, **kw))  # euler returns the trajectory tensor
        assert rec.step == nfe - 1, f"chunk {c}: {rec.step + 1} DiT calls, expected {nfe}"
        return out

    def pinyin(text_list: list[str], polyphone: bool = True) -> list[list[str]]:
        out = real_pinyin(text_list, polyphone)
        # called once per chunk, before odeint, so the chunk index is not advanced yet
        rec.chunks.append({"joined_text": text_list[0], "tokens": out[0]})
        return out

    def dit_pre(_m: nn.Module, _args: tuple[Any, ...], kw: dict[str, Any]) -> None:  # Any: hook arguments
        rec.step += 1
        c, s = rec.chunk, rec.step
        rec.put_once(f"c{c}.dit.step_cond", kw["cond"], "dit", "b n d")
        rec.put_once(f"c{c}.dit.text_ids", kw["text"], "dit", "b nt")
        rec.put(f"c{c}.dit.s{s:02d}.x", kw["x"], "dit", "b n d")
        rec.put(f"c{c}.dit.s{s:02d}.t", kw["time"].reshape(1), "dit", "b")
        rec.internal_calls = {}

    def dit_post(_m: nn.Module, _args: tuple[Any, ...], _kw: dict[str, Any], out: Tensor) -> None:  # Any: hook args
        c, s = rec.chunk, rec.step
        cond, uncond = torch.chunk(out, 2, dim=0)
        rec.put(f"c{c}.dit.s{s:02d}.out_cond", cond, "dit", "b n d")
        rec.put(f"c{c}.dit.s{s:02d}.out_uncond", uncond, "dit", "b n d")

    def internal(name: str, axes: str, every_step: bool = True) -> Callable[..., None]:
        def hook(_m: nn.Module, _args: tuple[Any, ...], out: Tensor) -> None:  # Any: hook arguments
            if rec.chunk != 0 or (every_step and rec.step not in rec.internal_steps):
                return
            # text_embed and input_embed run once for cond and once for uncond
            n = rec.internal_calls.get(name, 0)
            rec.internal_calls[name] = n + 1
            suffix = {0: "", 1: "_uncond"}[n] if name in ("text_embed", "input_embed") else ""
            prefix = "c0.dit_internal" if not every_step else f"c0.dit_internal.s{rec.step:02d}"
            rec.put(f"{prefix}.{name}{suffix}", out, "dit_internal", axes)

        return hook

    def mel_pre(_m: nn.Module, args: tuple[Tensor, ...]) -> None:
        rec.put_once("audio.ref", args[0], "audio", "b nw")

    def mel_post(_m: nn.Module, _args: tuple[Tensor, ...], out: Tensor) -> None:
        rec.put_once("mel.ref", out, "mel", "b d n")

    def backbone_pre(_m: nn.Module, args: tuple[Tensor, ...]) -> None:
        rec.put(f"c{rec.chunk}.mel_out", args[0], "mel_out", "b d n")

    def backbone_post(_m: nn.Module, _args: tuple[Tensor, ...], out: Tensor) -> None:
        rec.put(f"c{rec.chunk}.vocoder.backbone", out, "vocoder", "b n d")

    def head_out_post(_m: nn.Module, _args: tuple[Tensor, ...], out: Tensor) -> None:
        rec.put(f"c{rec.chunk}.vocoder.head_out", out, "vocoder", "b n f")

    # internal hooks go on the DiT's own submodules
    hooks.append(dit.text_embed.register_forward_hook(internal("text_embed", "b n d", every_step=False)))
    hooks.append(dit.time_embed.register_forward_hook(internal("time_embed", "b d")))
    hooks.append(dit.input_embed.register_forward_hook(internal("input_embed", "b n d")))
    for i, block in enumerate(dit.transformer_blocks):
        hooks.append(block.register_forward_hook(internal(f"block{i:02d}", "b n d")))
    hooks.append(dit.norm_out.register_forward_hook(internal("norm_out", "b n d")))
    hooks.append(dit.register_forward_pre_hook(dit_pre, with_kwargs=True))
    hooks.append(dit.register_forward_hook(dit_post, with_kwargs=True))
    hooks.append(model.mel_spec.register_forward_pre_hook(mel_pre))
    hooks.append(model.mel_spec.register_forward_hook(mel_post))
    hooks.append(vocoder.backbone.register_forward_pre_hook(backbone_pre))
    hooks.append(vocoder.backbone.register_forward_hook(backbone_post))
    head_out = getattr(vocoder.head, "out")  # ISTFTHead.out; the head is typed as the FourierHead base
    assert isinstance(head_out, nn.Linear)
    hooks.append(head_out.register_forward_hook(head_out_post))
    cfm_module.odeint = odeint  # pyright: ignore[reportAttributeAccessIssue]  # temporary patch, restored below
    ui.convert_char_to_pinyin = pinyin
    ui.ThreadPoolExecutor = lambda: SyncExecutor(rec)  # pyright: ignore[reportAttributeAccessIssue]  # as above
    try:
        yield
    finally:
        cfm_module.odeint = real_odeint
        ui.convert_char_to_pinyin = real_pinyin
        ui.ThreadPoolExecutor = real_executor
        for h in hooks:
            h.remove()


def run_case(case: Case, model: CFM, vocoder: Vocos, out: Path, threads: int) -> Recorder:
    rec = Recorder(internal_steps={0, case.nfe // 2})
    src = inspect.getsource(ui.infer_process)
    for line in COPIED_LINES:
        assert line in src, f"infer_process changed; update make_golden.py (R9): {line}"

    input_path = make_input(case, out)
    in_wav, in_sr = sf.read(str(input_path), dtype="float32", always_2d=True)
    rec.put("input.wav", in_wav.T, "input", "c nw")
    rec.values["input"] = {"sample_rate": in_sr, "ref_text": case.ref_text, "gen_text": case.gen_text}

    prep_path, ref_text = ui.preprocess_ref_audio_text(str(input_path), case.ref_text, show_info=lambda _s: None)
    audio_np, sr = sf.read(prep_path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio_np.T.copy())  # (channels, samples), as torchaudio.load gives it
    rec.put("prep.audio", audio, "prep", "c nw")
    rec.values["prep"] = {"sample_rate": sr, "ref_text": ref_text}

    speed = case.speed
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * speed)
    gen_text_batches = ui.chunk_text(case.gen_text, max_chars=max_chars)
    if case.expect_chunks is not None:
        assert len(gen_text_batches) == case.expect_chunks, [len(c.encode()) for c in gen_text_batches]
    ref_text_spaced = ref_text + " " if len(ref_text[-1].encode("utf-8")) == 1 else ref_text
    rec.values["text"] = {"max_chars": max_chars, "chunks": gen_text_batches, "ref_text_spaced": ref_text_spaced}

    mono = audio.mean(dim=0, keepdim=True) if audio.shape[0] > 1 else audio
    rms = float(torch.sqrt(torch.mean(torch.square(mono))))
    rec.values["audio"] = {"rms": rms, "raised": rms < ui.target_rms}

    torch.set_num_threads(threads)
    seed_everything(case.seed)
    with recording(model, vocoder, rec, case.nfe), torch.inference_mode():
        result = next(
            ui.infer_batch_process(
                (audio, sr),
                ref_text,
                gen_text_batches,
                model,
                vocoder,
                mel_spec_type="vocos",
                progress=None,
                nfe_step=case.nfe,
                cfg_strength=case.cfg,
                sway_sampling_coef=case.sway,
                speed=speed,
                fix_duration=case.fix_duration,
                device="cpu",
            )
        )
    final_wave = result[0]
    assert final_wave is not None
    rec.put("final.wav", np.asarray(final_wave, dtype=np.float32), "final", "nw")
    assert len(rec.chunks) == len(gen_text_batches) == rec.chunk + 1

    ref_audio_len = rec.tensors["audio.ref"].tensor.shape[-1] // HOP
    for i, (chunk, text) in enumerate(zip(rec.chunks, gen_text_batches)):
        chunk["text"] = text
        chunk["local_speed"] = speed * 0.3 if len(text.encode("utf-8")) < 10 else speed
        chunk["ref_audio_len"] = ref_audio_len
        assert rec.tensors[f"c{i}.mel_out"].tensor.shape[-1] == chunk["duration"] - ref_audio_len
    rec.values["mel"] = {"n_ref": rec.tensors["mel.ref"].tensor.shape[-1]}
    return rec


def write_case(case: Case, rec: Recorder, out: Path, threads: int, elapsed: float) -> dict[str, str]:
    for old in out.glob("*.npy"):
        old.unlink()
    hashes: dict[str, str] = {}
    entries = []
    for name, e in rec.tensors.items():
        np.save(out / f"{name}.npy", e.tensor)
        digest = hashlib.sha256(e.tensor.tobytes()).hexdigest()
        hashes[name] = digest
        entries.append(
            {
                "name": name,
                "file": f"{name}.npy",
                "stage": e.stage,
                "shape": list(e.tensor.shape),
                "axes": e.axes,
                "dtype": str(e.tensor.dtype),
                "sha256": digest,
            }
        )
    sf.write(str(out / "final.wav"), rec.tensors["final.wav"].tensor, SR, subtype="FLOAT")
    manifest = {
        "case": case.name,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        # torch.randn is vectorized only on AVX2, so the noise depends on the CPU that made it (plan 04 D1)
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "os": platform.platform(),
        "device": "cpu",
        "dtype": "float32",
        "num_threads": threads,
        "elapsed_s": round(elapsed, 1),
        "settings": {
            "model": case.model,
            "nfe_step": case.nfe,
            "cfg_strength": case.cfg,
            "sway_sampling_coef": case.sway,
            "speed": case.speed,
            "fix_duration": case.fix_duration,
            "seed": case.seed,
            "target_rms": ui.target_rms,
            "cross_fade_duration": ui.cross_fade_duration,
            "dit_internal_steps": sorted(rec.internal_steps),
            "dit_internal_chunks": [0],
        },
        **rec.values,
        "chunks": rec.chunks,
        "tensors": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return hashes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", action="append", choices=sorted(CASES), default=[])
    p.add_argument("--all", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--init-tiny", action="store_true", help="make new tiny weights (overwrites the committed ones)")
    p.add_argument("--no-recheck", action="store_true", help="skip the second, determinism-check run")
    p.add_argument(
        "--out-root", type=Path, default=OUT_ROOT, help="write cases here (tiny weights are still read from testdata)"
    )
    args = p.parse_args()
    names: list[str] = sorted(CASES) if args.all else args.case
    if not names:
        p.error("give --case or --all")

    failed: list[str] = []
    models: dict[str, tuple[CFM, Vocos]] = {}
    for name in names:
        case = CASES[name]
        if case.model not in models:
            models[case.model] = load_models(case, args.init_tiny)
        model, vocoder = models[case.model]
        out = args.out_root / name
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rec = run_case(case, model, vocoder, out, args.threads)
        elapsed = time.time() - t0
        hashes = write_case(case, rec, out, args.threads, elapsed)
        size = sum(e.tensor.nbytes for e in rec.tensors.values()) / 2**20
        print(f"{name}: {len(rec.chunks)} chunk(s), {len(hashes)} tensors, {size:.0f} MiB, {elapsed:.0f} s", flush=True)
        if not args.no_recheck:
            again = run_case(case, model, vocoder, out, args.threads)
            diff = [
                k
                for k in hashes
                if k not in again.tensors or hashlib.sha256(again.tensors[k].tensor.tobytes()).hexdigest() != hashes[k]
            ]
            if diff or set(again.tensors) != set(hashes):
                failed.append(name)
                print(f"{name}: NOT DETERMINISTIC, {len(diff)} tensors differ, first: {diff[:5]}", flush=True)
            else:
                print(f"{name}: second run identical", flush=True)
    if failed:
        sys.exit(f"determinism check failed: {failed}")


if __name__ == "__main__":
    main()
