"""Self-tests of tools/webgpu/parity.py (plan 00, step 6); the same scenarios as web/test/parity/parity.test.ts.

  python tools/webgpu/parity_selftest.py           # tiny case, plus the CUDA smoke test where dumps exist

The smoke test runs the CUDA fp32 rerun saved by `measure_floor.py --dump` through the runner as if it
were a port. It passes by construction (the tolerances come from it), so it checks plumbing only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity import (  # noqa: E402
    GOLDEN_ROOT,
    GoldenCase,
    Result,
    checks_for,
    compare,
    compare_dir,
    load_tolerances,
    report,
)


CHECKS_TABLE = {  # the same table is in parity.test.ts
    "c0.dit.s03.out_cond": ["dit.out"],
    "c2.dit.s31.out_uncond": ["dit.out"],
    "c0.dit_internal.s16.block21": ["dit_internal.block"],
    "c0.dit_internal.text_embed_uncond": ["dit_internal.text_embed"],
    "c0.dit_internal.s00.input_embed": ["dit_internal.input_embed"],
    "c0.dit_internal.s16.norm_out": ["dit_internal.norm_out"],
    "mel.ref": ["mel.log_masked", "mel.linear"],
    "c1.vocoder.head_out": ["vocoder.head_out", "vocoder.stft_complex"],
    "c0.dit.text_ids": ["exact"],
    "c0.noise.y0": [],
    "c0.dit.s00.x": [],
    "c0.dit.step_cond": [],
    "final.wav": [],
}
REAL = ["en-short", "zh-short", "en-multi", "en-multi-fixdur", "en-prep", "en-epss16"]


def compare_all(g: GoldenCase, tol: dict, override: dict[str, np.ndarray] | None = None) -> list[Result]:
    override = override or {}
    out: list[Result] = []
    for name in g.entries:
        if checks_for(name):
            out.extend(compare(g, tol, name, override.get(name, g.tensor(name))))
    return out


def main() -> None:
    tol = load_tolerances()
    g = GoldenCase("tiny")
    failures: list[str] = []

    def check(cond: bool, what: str) -> None:
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            failures.append(what)

    for name, expected in CHECKS_TABLE.items():
        check(checks_for(name) == expected, f"checks_for({name})")
    missing = {c for name in g.entries for c in checks_for(name) if c != "exact" and c not in tol["checks"]}
    check(not missing, f"every check has a tolerance (missing: {sorted(missing)})")

    _text, stage, failed = report(compare_all(g, tol))
    check(failed == 0 and stage is None, "golden against itself passes")

    dit = g.tensor("c0.dit.s03.out_cond") * np.float32(1.01)
    text, stage, failed = report(compare_all(g, tol, {"c0.dit.s03.out_cond": dit}))
    check(failed == 1 and stage == "dit" and "c0.dit.s03.out_cond" in text.splitlines()[-1], "perturbed DiT -> dit")

    dit2 = g.tensor("c0.dit.s03.out_cond").copy()
    dit2.flat[0] += 10
    wav = g.tensor("c0.vocoder.wav").copy()
    wav[0] += 1
    mel = g.tensor("mel.ref") + np.float32(0.5)
    _t, stage, _f = report(compare_all(g, tol, {"c0.dit.s03.out_cond": dit2, "c0.vocoder.wav": wav, "mel.ref": mel}))
    check(stage == "mel", "earliest failing stage is mel")

    mel_t = np.ascontiguousarray(g.tensor("mel.ref").transpose(0, 2, 1))
    res = compare(g, tol, "mel.ref", mel_t)
    check(all(not r.passed and (r.error or "").startswith("shape") and not r.metrics for r in res), "transposed mel")

    wav_nan = g.tensor("c0.vocoder.wav").copy()
    wav_nan[100] = np.nan
    check(not compare(g, tol, "c0.vocoder.wav", wav_nan)[0].passed, "NaN fails")
    ids = g.tensor("c0.dit.text_ids").copy()
    ids.flat[0] += 1
    check(not compare(g, tol, "c0.dit.text_ids", ids)[0].passed, "wrong token id fails")
    try:
        compare(g, tol, "c0.noise.y0", g.tensor("c0.noise.y0"))
        check(False, "input refused")
    except ValueError:
        check(True, "input refused")

    for case in REAL:
        dump = GOLDEN_ROOT / case / "cuda"
        if not dump.exists():
            print(f"skip smoke {case}: no {dump}")
            continue
        results = compare_dir(GoldenCase(case), tol, dump)
        text, stage, failed = report(results)
        check(failed == 0 and len(results) > 50, f"smoke {case}: {len(results)} comparisons")
        if failed:
            print(text)

    print(f"{len(failures)} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
