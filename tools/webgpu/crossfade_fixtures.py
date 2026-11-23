"""Plan 04 step 5: upstream's cross-fade on cases the golden data never reaches, from its own code.

  .venv313/Scripts/python.exe tools/webgpu/crossfade_fixtures.py

Runs the real infer_batch_process with plan_fixtures.py's stubs and a vocoder that returns preset waves, one
per chunk in order, so the cross-fade under test is upstream's. The reference is loud enough that no output
rescale applies. Writes testdata/webgpu/pipeline/crossfade_cases.json, with the waves as base64 float32 and
the result as base64 of whatever dtype upstream returned (float64 once any fade has run).
"""

from __future__ import annotations

import base64
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

import f5_tts.infer.utils_infer as ui
import f5_tts.model.cfm as cfm_module
from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer


sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_fixtures import SR, VOCAB, StubTransformer, SyncExecutor  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "testdata/webgpu/pipeline/crossfade_cases.json"

# name, wave lengths, cross_fade_duration
CASES: list[tuple[str, list[int], float]] = [
    ("default-fade", [4000, 4500, 3700], 0.15),
    ("shorter-than-fade", [900, 300, 800], 0.02),
    ("first-shorter-than-fade", [200, 900], 0.02),
    ("zero", [500, 600], 0.0),
    ("negative", [500, 600], -1.0),
    ("two-samples", [500, 600, 700], 0.0001),
    ("one-sample", [500, 600], 1 / 24000),
    ("single-chunk", [700], 0.15),
    ("fractional-samples", [900, 800], 0.02004),  # 480.96 samples: truncated to 480
    ("five-chunks", [600, 700, 650, 900, 400], 0.01),
]


class PresetVocoder:
    def __init__(self, waves: list[np.ndarray]) -> None:
        self.waves = list(waves)

    def decode(self, mel: Tensor) -> Tensor:
        return torch.from_numpy(self.waves.pop(0)).unsqueeze(0)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def run_case(model: CFM, lengths: list[int], fade: float, rng: np.random.Generator) -> dict[str, Any]:  # Any: json
    waves = [(rng.standard_normal(n) * 0.2).astype(np.float32) for n in lengths]
    audio = torch.sin(torch.arange(SR * 2, dtype=torch.float32) * 0.05).unsqueeze(0) * 0.3
    real_odeint, real_executor = cfm_module.odeint, ui.ThreadPoolExecutor

    def odeint(fn: Callable[..., Tensor], y0: Tensor, t: Tensor, **kw: Any) -> Tensor:  # Any: odeint options
        return y0.unsqueeze(0)

    cfm_module.odeint = odeint
    ui.ThreadPoolExecutor = SyncExecutor
    try:
        vocoder = PresetVocoder(waves)
        texts = [f"Chunk number {i}." for i in range(len(lengths))]
        result = next(
            ui.infer_batch_process(
                (audio, SR), "Hello there.", texts, model, vocoder, progress=None, device="cpu",
                cross_fade_duration=fade,
            )
        )  # fmt: skip
        assert len(result) == 3, "not streaming: (wave, sample rate, spectrogram)"
        wave, sr, _ = result
        assert not vocoder.waves and sr == SR
    finally:
        cfm_module.odeint, ui.ThreadPoolExecutor = real_odeint, real_executor
    assert isinstance(wave, np.ndarray)
    return {"fade": fade, "waves": [b64(w) for w in waves], "dtype": str(wave.dtype), "result": b64(wave)}


def main() -> None:
    vocab, _ = get_tokenizer(str(VOCAB), "custom")
    assert isinstance(vocab, dict)
    model = CFM(transformer=StubTransformer(), mel_spec_kwargs={"mel_spec_type": "vocos"}, vocab_char_map=vocab)
    rng = np.random.default_rng(0)
    cases = {name: run_case(model, lengths, fade, rng) for name, lengths, fade in CASES}
    OUT.write_text(json.dumps(cases, indent=1) + "\n", encoding="utf-8")
    for name, c in cases.items():
        print(name, c["dtype"], len(base64.b64decode(c["result"])))


if __name__ == "__main__":
    main()
