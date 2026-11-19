"""Parity runner, Python twin of web/test/parity/ (plan 00, step 6).

Compares a port's tensors with the golden data, using testdata/webgpu/tolerances.json. Plans 01
and 02 use it to check an export (for example in onnxruntime) before anything runs in a browser.

  python tools/webgpu/parity.py --case en-short --actual DIR

DIR holds .npy files named like the golden tensors (`c0.dit.s05.out_cond.npy`). Golden inputs in
DIR are ignored. Exit code 1 if any check fails; the earliest failing stage is printed last.
The check names, metrics and pass rules must stay the same as in web/test/parity/compare.ts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
GOLDEN_ROOT = REPO / "testdata/webgpu"

STAGE_ORDER = [
    "input",
    "prep",
    "text",
    "audio",
    "mel",
    "plan",
    "noise",
    "dit",
    "dit_internal",
    "mel_out",
    "vocoder",
    "final",
]
INFORMATIONAL = {"audio.resample", "vocoder.head_out"}
LOG_MEL_MIN = math.log(1e-4)


def checks_for(name: str) -> list[str]:
    """Which tolerance checks apply to a golden tensor; inputs have none. See compare.ts checksFor."""
    n = re.sub(r"^c\d+\.", "", name)
    if n == "plan.t":
        return ["plan.t"]
    if name == "audio.ref":
        return ["audio.resample"]
    if name == "mel.ref":
        return ["mel.log_masked", "mel.linear"]
    if n == "dit.text_ids":
        return ["exact"]
    if re.fullmatch(r"dit\.s\d+\.out_(cond|uncond)", n):
        return ["dit.out"]
    m = re.fullmatch(r"dit_internal\.(?:s\d+\.)?([a-z_]+?)(\d+)?(?:_uncond)?", n)
    if m:
        return ["dit_internal.block" if m.group(1) == "block" else f"dit_internal.{m.group(1)}"]
    if n == "mel_out":
        return ["mel_out.free_running"]
    if n == "vocoder.backbone":
        return ["vocoder.backbone"]
    if n == "vocoder.head_out":
        return ["vocoder.head_out", "vocoder.stft_complex"]
    if n == "vocoder.wav":
        return ["vocoder.wav"]
    return []


def metrics(got: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | None]:
    a = got.astype(np.float64)
    b = ref.astype(np.float64)
    if mask is not None:
        a, b = a[mask], b[mask]
    d = a - b
    nb = float(np.linalg.norm(b))
    small = nb < 1e-6 * b.size
    na = float(np.linalg.norm(a))
    return {
        "max_abs": float(np.abs(d).max()) if d.size else 0.0,  # NaN propagates and fails below
        "rel_l2": None if small else float(np.linalg.norm(d) / nb),
        "cos": None if small or na == 0 else float((a * b).sum() / (na * nb)),
    }


def complex_stft(head_out: np.ndarray) -> np.ndarray:
    """Vocos ISTFTHead: first half log magnitude (exp, clipped at 100), second half phase."""
    mag_log, phase = np.split(head_out.astype(np.float64), 2, axis=-1)
    mag = np.minimum(np.exp(mag_log), 100.0)
    return np.concatenate([mag * np.cos(phase), mag * np.sin(phase)], axis=-1)


@dataclass
class Result:
    stage: str
    name: str
    check: str
    passed: bool
    informational: bool
    error: str | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    tolerance: dict[str, float] = field(default_factory=dict)


class GoldenCase:
    def __init__(self, case: str, root: Path = GOLDEN_ROOT) -> None:
        self.dir = root / case
        self.manifest: dict[str, Any] = json.loads(
            (self.dir / "manifest.json").read_text(encoding="utf-8")
        )  # Any: json
        self.entries: dict[str, dict[str, Any]] = {t["name"]: t for t in self.manifest["tensors"]}  # Any: json
        self._cache: dict[str, np.ndarray] = {}

    def tensor(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = np.load(self.dir / self.entries[name]["file"])
        return self._cache[name]


def load_tolerances(root: Path = GOLDEN_ROOT) -> dict[str, Any]:  # Any: json
    return json.loads((root / "tolerances.json").read_text())


def compare(golden: GoldenCase, tolerances: dict[str, Any], name: str, actual: np.ndarray) -> list[Result]:  # Any: json
    entry = golden.entries.get(name)
    if entry is None:
        raise KeyError(f"{golden.manifest['case']}: no golden tensor {name}")
    checks = checks_for(name)
    if not checks:
        raise ValueError(f"{name} is an input; there is no check for it")
    stage = entry["stage"]
    if list(actual.shape) != entry["shape"]:
        err = f"shape {list(actual.shape)} but golden is {entry['shape']} ({entry['axes']})"
        return [Result(stage, name, c, False, c in INFORMATIONAL, err) for c in checks]
    if str(actual.dtype) != entry["dtype"]:
        err = f"dtype {actual.dtype} but golden is {entry['dtype']}"
        return [Result(stage, name, c, False, c in INFORMATIONAL, err) for c in checks]
    ref = golden.tensor(name)
    out: list[Result] = []
    for check in checks:
        info = check in INFORMATIONAL
        if check == "exact":
            same = bool(np.array_equal(actual, ref))
            out.append(Result(stage, name, check, same, info, None if same else "values differ"))
            continue
        if check == "mel.log_masked":
            m = metrics(actual, ref, mask=ref > LOG_MEL_MIN)
        elif check == "mel.linear":
            m = metrics(np.exp(actual.astype(np.float64)), np.exp(ref.astype(np.float64)))
        elif check == "vocoder.stft_complex":
            m = metrics(complex_stft(actual), complex_stft(ref))
        else:
            m = metrics(actual, ref)
        tol = tolerances["checks"][check]
        if golden.manifest["case"] == "tiny":
            tol = {**tol, **tol.get("tiny", {})}  # tiny's own tolerance where its floor is higher (R5)
        rel_tol, abs_tol = tol.get("rel_l2"), tol.get("max_abs")
        # near-zero golden: relative L2 is undefined, so max abs decides, against rel_l2 if there is no max_abs
        abs_limit = abs_tol if abs_tol is not None else (rel_tol if m["rel_l2"] is None else None)
        max_abs = m["max_abs"]
        assert max_abs is not None
        ok_abs = abs_limit is None or max_abs <= abs_limit
        ok_rel = rel_tol is None or m["rel_l2"] is None or m["rel_l2"] <= rel_tol
        passed = not math.isnan(max_abs) and ok_abs and ok_rel
        tolerance = {k: v for k, v in (("rel_l2", rel_tol), ("max_abs", abs_tol)) if v is not None}
        out.append(Result(stage, name, check, passed or info, info, None, m, tolerance))
    return out


def report(results: list[Result]) -> tuple[str, str | None, int]:
    """One line per comparison; the earliest failing stage (pipeline order) last. Returns text, stage, failures."""

    def fmt(x: float | None) -> str:
        return "-" if x is None else f"{x:.2e}"

    lines = []
    for r in results:
        status = "INFO" if r.informational else "ok  " if r.passed else "FAIL"
        detail = r.error or (
            f"relL2 {fmt(r.metrics.get('rel_l2'))} (tol {fmt(r.tolerance.get('rel_l2'))})  "
            f"maxAbs {fmt(r.metrics.get('max_abs'))} (tol {fmt(r.tolerance.get('max_abs'))})"
        )
        lines.append(f"{status} {r.stage:<12} {r.check:<26} {r.name:<40} {detail}")
    failures = sorted((r for r in results if not r.passed), key=lambda r: STAGE_ORDER.index(r.stage))
    if failures:
        f = failures[0]
        lines.append(f"FIRST FAILING STAGE: {f.stage} ({f.check}, {f.name}); {len(failures)} failure(s)")
    else:
        lines.append(f"all {len(results)} comparisons pass")
    return "\n".join(lines), failures[0].stage if failures else None, len(failures)


def compare_dir(golden: GoldenCase, tolerances: dict[str, Any], actual_dir: Path) -> list[Result]:  # Any: json
    results: list[Result] = []
    for path in sorted(actual_dir.glob("*.npy")):
        name = path.stem
        if name in golden.entries and checks_for(name):
            results.extend(compare(golden, tolerances, name, np.load(path)))
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True)
    p.add_argument("--actual", type=Path, required=True, help="folder of .npy files named like the golden tensors")
    p.add_argument("--quiet", action="store_true", help="print failures and the summary only")
    args = p.parse_args()
    results = compare_dir(GoldenCase(args.case), load_tolerances(), args.actual)
    if not results:
        sys.exit(f"no comparable tensors in {args.actual}")
    text, _stage, failed = report(results)
    if args.quiet:
        text = "\n".join(line for line in text.splitlines() if not line.startswith(("ok  ", "INFO")))
    print(text)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
