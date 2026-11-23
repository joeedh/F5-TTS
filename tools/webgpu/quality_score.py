"""Plan 04 step 7 (D5): scores the PyTorch and port generations with upstream's eval, and compares them.

  .venv313/Scripts/python.exe tools/webgpu/quality_score.py --lang en --task wer
  .venv313/Scripts/python.exe tools/webgpu/quality_score.py --lang en --task sim
  .venv313/Scripts/python.exe tools/webgpu/quality_score.py --lang en --report

On Windows, set FFMPEG_DLL_DIRS first (see dll_dirs.py). Each task calls upstream's own get_seed_tts_test,
run_asr_wer and run_sim in this process, on GPU 0: what eval_seedtts_testset.py does with one GPU, without its
process pool. Writes data/seedtts_testset/gen/<side>/<lang>/_<task>_results.jsonl in upstream's format.
--report reads them and prints the means, the paired differences and the utterances that differ most.
D5 passes if the port's mean WER is within 0.5 points (0.005) of PyTorch's and its mean SIM within 0.01.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from dll_dirs import register_cuda12, register_ffmpeg  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/seedtts_testset"
WAVLM = REPO.parent / "checkpoints/UniSpeech/wavlm_large_finetune.pth"
SIDES = ["torch", "webgpu"]
WER_BOUND = 0.005
SIM_BOUND = 0.01


def stub_torchaudio_for_s3prl() -> None:
    """s3prl's hubconf imports every model it has, and some call torchaudio APIs removed in 2.9 at import
    time: set_audio_backend (backend selection is gone, so a no-op is faithful) and the sox_effects module
    (its one function raises if called; WavLM never calls it)."""
    import types

    import torchaudio

    if not hasattr(torchaudio, "set_audio_backend"):
        setattr(torchaudio, "set_audio_backend", lambda *_a, **_k: None)
    if "torchaudio.sox_effects" not in sys.modules:
        sox = types.ModuleType("torchaudio.sox_effects")

        def apply_effects_tensor(*_a: object, **_k: object) -> None:
            raise NotImplementedError("torchaudio.sox_effects was removed; stubbed only so s3prl imports")

        setattr(sox, "apply_effects_tensor", apply_effects_tensor)
        sys.modules["torchaudio.sox_effects"] = sox


def score(lang: str, task: str) -> None:
    register_ffmpeg()  # run_sim and the ASR models read the wavs through torchaudio or av
    register_cuda12()  # faster-whisper on the GPU
    stub_torchaudio_for_s3prl()
    from f5_tts.eval.utils_eval import get_seed_tts_test, run_asr_wer, run_sim

    for side in SIDES:
        gen = DATA / "gen" / side / lang
        test_set = get_seed_tts_test(str(DATA / lang / "meta.lst"), str(gen), [0])[0][1]
        print(f"{side}/{lang}: {len(test_set)} utterances", flush=True)
        if task == "wer":
            results = run_asr_wer((0, lang, test_set, ""))
        else:
            assert WAVLM.exists(), f"needs {WAVLM}"
            results = run_sim((0, test_set, str(WAVLM)))
        lines = [json.dumps(r, ensure_ascii=False) for r in results]
        mean = float(np.mean([float(r[task]) for r in results]))
        (gen / f"_{task}_results.jsonl").write_text("\n".join(lines) + f"\n\n{task.upper()}: {mean:.5f}\n", "utf-8")
        print(f"{side}/{lang} {task}: {mean:.5f}", flush=True)


def load(lang: str, side: str, task: str) -> dict[str, dict[str, Any]]:  # Any: a result row
    path = DATA / "gen" / side / lang / f"_{task}_results.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln.startswith("{")]
    return {r["wav"]: r for r in rows}


def report(lang: str) -> None:
    for task, bound in (("wer", WER_BOUND), ("sim", SIM_BOUND)):
        t, w = load(lang, "torch", task), load(lang, "webgpu", task)
        if not t or not w:
            print(f"{lang} {task}: not scored yet")
            continue
        assert t.keys() == w.keys(), f"{lang} {task}: the two sides scored different utterances"
        keys = sorted(t)
        a = np.array([t[k][task] for k in keys])
        b = np.array([w[k][task] for k in keys])
        diff = b - a
        verdict = "PASS" if abs(diff.mean()) <= bound else "FAIL"
        print(
            f"{lang} {task}: torch {a.mean():.5f}, webgpu {b.mean():.5f}, difference {diff.mean():+.5f}"
            f" (bound {bound}) {verdict}; {int((diff != 0).sum())} of {len(keys)} utterances differ"
        )
        for i in np.argsort(-np.abs(diff))[:5]:
            k = keys[i]
            extra = f"\n    torch:  {t[k]['hypo']}\n    webgpu: {w[k]['hypo']}" if task == "wer" else ""
            print(f"  {k}: torch {a[i]:.4f}, webgpu {b[i]:.4f}{extra}")
    print(f"versions: faster-whisper {version('faster-whisper')}, funasr {version('funasr')}, jiwer {version('jiwer')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", choices=["en", "zh"], required=True)
    p.add_argument("--task", choices=["wer", "sim"])
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    if args.task:
        score(args.lang, args.task)
    if args.report:
        report(args.lang)


if __name__ == "__main__":
    main()
