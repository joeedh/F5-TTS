"""Plan 04 step 7 (D5): the PyTorch side of the quality comparison, paired with the port utterance by utterance.

  .venv313/Scripts/python.exe tools/webgpu/quality_torch.py --lang en

On Windows, set FFMPEG_DLL_DIRS first (see dll_dirs.py).

Runs upstream's own preprocess_ref_audio_text and infer_process on CUDA in fp32 (TF32 off), with two changes
that pair it with the port:
- chunks run in order (plan_fixtures.py's SyncExecutor);
- the starting noise is drawn on the CPU from torch.manual_seed(seed), where CFM.sample would draw it on the
  GPU. CUDA's generator is unrelated to the CPU's, which the port reproduces (plan 04, D1).
Reads data/seedtts_testset/quality_subset.json (quality_set.py) and writes
data/seedtts_testset/gen/torch/<lang>/<utt>.wav (float32) and timing.json.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import soundfile as sf
import torch
from torch import Tensor

import f5_tts.infer.utils_infer as ui
import f5_tts.model.cfm as cfm_module


sys.path.insert(0, str(Path(__file__).resolve().parent))
from dll_dirs import register_ffmpeg  # noqa: E402
from make_golden import CASES, load_models  # noqa: E402
from plan_fixtures import SyncExecutor  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/seedtts_testset"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", choices=["en", "zh"], required=True)
    p.add_argument("--limit", type=int, help="only the first utterances")
    args = p.parse_args()
    assert torch.cuda.is_available(), "needs CUDA (run in .venv313)"
    register_ffmpeg()  # infer_process reads the reference with torchaudio.load
    if isinstance(sys.stdout, io.TextIOWrapper):  # upstream prints the texts; a Windows pipe is cp1252, with no CJK
        sys.stdout.reconfigure(encoding="utf-8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = torch.device("cuda")

    rows = json.loads((DATA / "quality_subset.json").read_text(encoding="utf-8"))[args.lang][: args.limit]
    model, vocoder = load_models(CASES["en-short"], init_tiny=False)
    model, vocoder = model.to(dev).eval(), vocoder.to(dev).eval()
    out = DATA / "gen/torch" / args.lang
    out.mkdir(parents=True, exist_ok=True)

    real_odeint, real_executor = cfm_module.odeint, ui.ThreadPoolExecutor

    def odeint(fn: Callable[..., Tensor], y0: Tensor, t: Tensor, **kw: Any) -> Tensor:  # Any: odeint options
        cpu_noise = torch.randn(tuple(y0.shape), dtype=torch.float32)  # the CPU generator, as the port's noise
        return cast(Tensor, real_odeint(fn, cpu_noise.to(y0.device), t, **kw))

    timing: dict[str, float] = {}
    cfm_module.odeint = odeint
    ui.ThreadPoolExecutor = SyncExecutor
    try:
        for k, row in enumerate(rows):
            t0 = time.perf_counter()
            prompt = str(DATA / row["prompt_wav"])
            ref_path, ref_text = ui.preprocess_ref_audio_text(prompt, row["prompt_text"], show_info=lambda _s: None)
            torch.manual_seed(row["seed"])
            with torch.inference_mode():
                wave, sr, _ = ui.infer_process(
                    ref_path, ref_text, row["target_text"], model, vocoder, show_info=lambda _s: None,
                    progress=None, device="cuda",
                )  # fmt: skip
            assert wave is not None and sr == 24000
            sf.write(str(out / f"{row['utt']}.wav"), wave, sr, subtype="FLOAT")
            timing[row["utt"]] = round(time.perf_counter() - t0, 2)
            print(f"{k + 1}/{len(rows)} {row['utt']}: {len(wave) / sr:.2f} s in {timing[row['utt']]} s", flush=True)
    finally:
        cfm_module.odeint, ui.ThreadPoolExecutor = real_odeint, real_executor
    (out / "timing.json").write_text(json.dumps(timing, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
