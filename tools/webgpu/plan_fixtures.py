"""Plan 04 step 1: chunk plans for cases the golden data never reaches, from upstream's own code.

  .venv313\\Scripts\\python.exe tools/webgpu/plan_fixtures.py

Runs the real infer_batch_process and CFM.sample with a stub transformer and vocoder, and wraps odeint to
record each chunk's final duration (y0's length) and time grid. Chunks run in order, as in make_golden.py.
Writes testdata/webgpu/pipeline/plan_cases.json for web/test/pipeline/plan.test.ts.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import f5_tts.infer.utils_infer as ui
import f5_tts.model.cfm as cfm_module
from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "testdata/webgpu/pipeline/plan_cases.json"
VOCAB = REPO / "src/f5_tts/infer/examples/vocab.txt"
SR = 24000

EN_REF = "Some call me nature, others call me mother nature."
EN_LONG = (
    "This sentence is long enough that its tokens outnumber the frames a one second reference would give it,"
    " so the minimum duration rule decides how long the chunk is, not the speaking rate of the reference."
)
ZH_REF = "对，这就是我，万人敬仰的太乙真人。"
ZH_GEN = "突然，身边一阵笑声。我看着他们，意气风发地挺直了胸膛。"

# name, reference seconds, reference text, chunks, settings
CASES: list[tuple[str, float, str, list[str], dict[str, Any]]] = [  # Any: infer_batch_process settings
    ("min-duration", 1.0, EN_LONG, [EN_LONG], {}),
    ("short-chunk", 5.0, EN_REF, ["Hi there"], {}),
    ("ten-bytes", 5.0, EN_REF, ["Hi there!!"], {}),
    ("zh-twelve-bytes", 5.0, EN_REF, ["你好吗？"], {}),  # 4 UTF-16 units, 12 bytes: no slow-down
    ("speed", 5.0, EN_REF, ["I don't really care what you call me."], {"speed": 0.7}),
    ("fix-single", 5.0, EN_REF, ["I don't really care what you call me."], {"fix_duration": 12.3}),
    (
        "fix-multi",
        4.3,
        EN_REF,
        ["I don't really care what you call me.", "I've been a silent spectator.", "Yes."],
        {"fix_duration": 20.0, "cross_fade_duration": 0.25},
    ),
    ("nfe7-epss", 5.0, EN_REF, ["Hello there, world."], {"nfe_step": 7}),
    ("nfe20-linspace", 5.0, EN_REF, ["Hello there, world."], {"nfe_step": 20}),
    ("no-sway", 5.0, EN_REF, ["Hello there, world."], {"sway_sampling_coef": None}),
    ("sway-positive", 5.0, EN_REF, ["Hello there, world."], {"sway_sampling_coef": 0.4, "nfe_step": 16}),
    ("zh", 3.7, ZH_REF, [ZH_GEN], {}),
    ("odd-samples", 5.123, EN_REF, ["Hello there, world."], {}),
]


class StubTransformer(nn.Module):
    """Never called: odeint is replaced. It only gives CFM a parameter for its dtype and device."""

    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.dim = 4

    def clear_cache(self) -> None:
        pass


class StubVocoder:
    def decode(self, mel: Tensor) -> Tensor:
        return torch.zeros(1, (mel.shape[-1] - 1) * 256)


class SyncExecutor:
    """Runs submitted chunks at once, in order, as make_golden.py does."""

    def __enter__(self) -> SyncExecutor:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future[Any]:  # Any: the executor's interface
        f: Future[Any] = Future()  # Any: as above
        f.set_result(fn(*args))
        return f


def run_case(
    model: CFM, seconds: float, ref_text: str, chunks: list[str], settings: dict[str, Any]
) -> dict[str, Any]:  # Any: json
    samples = int(round(seconds * SR))
    audio = torch.sin(torch.arange(samples, dtype=torch.float32) * 0.05).unsqueeze(0) * 0.3
    got: list[dict[str, Any]] = []  # Any: json
    real_odeint = cfm_module.odeint

    def odeint(fn: Callable[..., Tensor], y0: Tensor, t: Tensor, **kw: Any) -> Tensor:  # Any: odeint options
        got.append({"duration": int(y0.shape[1]), "t": [float(x) for x in t.tolist()]})
        return y0.unsqueeze(0)

    real_sample = model.sample
    asked: list[dict[str, Any]] = []  # Any: json

    def sample(**kw: Any) -> Any:  # Any: CFM.sample arguments and result
        asked.append({"duration_asked": int(kw["duration"]), "tokens": len(kw["text"][0])})
        return real_sample(**kw)

    model.sample = sample  # pyright: ignore[reportAttributeAccessIssue]  # test double
    real_executor = ui.ThreadPoolExecutor
    cfm_module.odeint = odeint
    ui.ThreadPoolExecutor = SyncExecutor
    try:
        next(
            ui.infer_batch_process(
                (audio, SR), ref_text, chunks, model, StubVocoder(), progress=None, device="cpu", **settings
            )
        )
    finally:
        cfm_module.odeint = real_odeint
        ui.ThreadPoolExecutor = real_executor
        model.sample = real_sample
    assert len(got) == len(chunks) == len(asked)
    got = [{**a, **g} for a, g in zip(asked, got, strict=True)]
    return {"ref_samples": samples, "ref_text": ref_text, "chunks": chunks, "settings": settings, "results": got}


def main() -> None:
    vocab, _ = get_tokenizer(str(VOCAB), "custom")
    assert isinstance(vocab, dict)
    model = CFM(transformer=StubTransformer(), mel_spec_kwargs={"mel_spec_type": "vocos"}, vocab_char_map=vocab)
    model.eval()
    cases = {name: run_case(model, s, ref, ch, st) for name, s, ref, ch, st in CASES}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, c in cases.items():
        print(name, [(r["duration_asked"], r["tokens"], r["duration"], len(r["t"])) for r in c["results"]])


if __name__ == "__main__":
    main()
