"""Plan 02 step 3: run the exported Vocos graph in onnxruntime (CPU) against the golden data.

Feeds each chunk's golden c*.mel_out to vocos_debug.onnx and compares:
  vocoder.backbone      the backbone output
  vocoder.head_out      the head's linear output (informational; its phase half is unstable where the
                        magnitude is tiny)
  vocoder.stft_complex  the complex spectrum built from head_out, which is what the inverse STFT uses

  .venv313\\Scripts\\python.exe tools/webgpu/check_vocos_ort.py --case tiny
  .venv313\\Scripts\\python.exe tools/webgpu/check_vocos_ort.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort  # pyright: ignore[reportMissingImports]  # no Python 3.15 wheel; run in .venv313


sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity import GOLDEN_ROOT, REPO, GoldenCase, Result, compare, load_tolerances, report  # noqa: E402


def graph(case: str) -> Path:
    folder = REPO / "testdata/webgpu/tiny/onnx" if case == "tiny" else REPO / "build/webgpu/vocos"
    return folder / "vocos_debug.onnx"


def check_case(name: str, threads: int) -> list[Result]:
    g = GoldenCase(name)
    tolerances = load_tolerances()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(graph(name)), opts, providers=["CPUExecutionProvider"])
    out: list[Result] = []
    for c in range(len(g.manifest["chunks"])):
        backbone, head_out = cast(list[np.ndarray], sess.run(None, {"mel": g.tensor(f"c{c}.mel_out")}))
        out += compare(g, tolerances, f"c{c}.vocoder.backbone", backbone)
        out += compare(g, tolerances, f"c{c}.vocoder.head_out", head_out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", action="append", default=[], help="default: every case with golden data")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--quiet", action="store_true", help="print failures and the summary only")
    args = p.parse_args()
    cases = args.case or sorted(d.name for d in GOLDEN_ROOT.iterdir() if (d / "manifest.json").exists())
    failed_total = 0
    for name in cases:
        results = check_case(name, args.threads)
        text, first, failed = report(results)
        failed_total += failed
        if args.quiet:
            text = "\n".join(line for line in text.splitlines() if not line.startswith(("ok", "INFO")))
        print(f"== {name}: {len(results)} comparisons, {failed} failed, first failing stage {first}\n{text}")
    sys.exit(1 if failed_total else 0)


if __name__ == "__main__":
    main()
