"""Plan 01 step 1: run the exported DiT graphs in onnxruntime (CPU) against the golden data.

Teacher-forced like measure_floor.py: every step of every chunk gets the golden x, t and step_cond.
Checks per step:
  dit.out     pred halves against the golden out_cond and out_uncond
  dit.v       the CFG velocity, against the same formula on the golden halves
  dit.x_next  the Euler update, against the next step's golden x (or mel_out after the last step)
and, with --debug, the internal outputs at the golden dit_internal steps of chunk 0.

  .venv313\\Scripts\\python.exe tools/webgpu/check_dit_ort.py --case tiny --debug
  .venv313\\Scripts\\python.exe tools/webgpu/check_dit_ort.py --attention mha --quiet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort  # pyright: ignore[reportMissingImports]  # no Python 3.15 wheel; run in .venv313


sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity import (  # noqa: E402
    GOLDEN_ROOT,
    REPO,
    GoldenCase,
    Result,
    compare,
    judge,
    load_tolerances,
    metrics,
    report,
    x_next_golden,
)


def onnx_dir(case: str) -> Path:
    return REPO / "testdata/webgpu/tiny/onnx" if case == "tiny" else REPO / "build/webgpu/dit/base"


def session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.add_session_config_entry("session.disable_prepacking", "1")  # no second copy of the 1.3 GB weights
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def run(sess: ort.InferenceSession, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """`run` is typed to also return sparse tensors, lists and maps; these graphs return dense arrays."""
    return cast(list[np.ndarray], sess.run(None, feeds))


def pad_ids(ids: np.ndarray, n: int) -> np.ndarray:
    """Cut or pad the golden (1, nt) ids to n with -1, as TextEmbedding does inside (plan 01 D1)."""
    out = np.full((1, n), -1, dtype=np.int32)
    k = min(n, ids.shape[1])
    out[:, :k] = ids[:, :k]
    return out


def check_case(
    name: str,
    attention: str,
    debug: bool,
    threads: int,
    tolerances: dict[str, Any],  # Any: json
    max_steps: int | None,
) -> list[Result]:
    g = GoldenCase(name)
    man = g.manifest
    folder = onnx_dir(name)
    suffix = "" if attention == "unfused" else f"_{attention}"
    text_sess = session(folder / "text.onnx", threads)
    # one step graph at a time: the debug graph's first three outputs are the normal graph's
    step_sess = session(folder / f"step_debug{suffix}.onnx" if debug else folder / f"step{suffix}.onnx", threads)
    debug_names = [o.name for o in step_sess.get_outputs()] if debug else []
    cfg = np.float32(man["settings"]["cfg_strength"])
    nfe = man["settings"]["nfe_step"]
    internal_steps = set(man["settings"]["dit_internal_steps"])
    out: list[Result] = []

    def derived(check: str, label: str, got: np.ndarray, ref: np.ndarray) -> None:
        if check in tolerances["checks"]:
            out.append(judge(g, tolerances, "dit", label, check, metrics(got, ref)))
        else:  # until measure_floor.py has measured it
            out.append(Result("dit", label, check, True, True, None, metrics(got, ref)))

    for c, chunk in enumerate(man["chunks"]):
        n = chunk["duration"]
        tc, tu = run(text_sess, {"text": pad_ids(g.tensor(f"c{c}.dit.text_ids"), n)})
        if c == 0:
            out += compare(g, tolerances, "c0.dit_internal.text_embed", tc)
            out += compare(g, tolerances, "c0.dit_internal.text_embed_uncond", tu)
        grid = g.tensor(f"c{c}.plan.t")
        step_cond = g.tensor(f"c{c}.dit.step_cond")
        for s in range(nfe if max_steps is None else min(nfe, max_steps)):
            p = f"c{c}.dit.s{s:02d}"
            feeds = {
                "x": g.tensor(f"{p}.x"),
                "step_cond": step_cond,
                "text_cond": tc,
                "text_uncond": tu,
                "time": g.tensor(f"{p}.t"),
                "cfg_strength": np.array([cfg], dtype=np.float32),
                "dt": np.array([grid[s + 1] - grid[s]], dtype=np.float32),
            }
            internal = debug and c == 0 and s in internal_steps
            res = run(step_sess, feeds)
            pred, v, x_next = res[0], res[1], res[2]
            out += compare(g, tolerances, f"{p}.out_cond", pred[:1])
            out += compare(g, tolerances, f"{p}.out_uncond", pred[1:])
            gc, gu = g.tensor(f"{p}.out_cond"), g.tensor(f"{p}.out_uncond")
            derived("dit.v", f"{p}.v", v, gc + (gc - gu) * cfg)
            start, ref = x_next_golden(g, c, s)
            derived("dit.x_next", f"{p}.x_next", x_next[:, start:, :], ref)
            if internal:
                for key, val in zip(debug_names[3:], res[3:]):
                    gname = f"c0.dit_internal.s{s:02d}.{key}"
                    if gname in g.entries:
                        out += compare(g, tolerances, gname, val)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", action="append", default=[], help="default: every case with golden data")
    p.add_argument("--attention", choices=["unfused", "mha", "onnx23"], default="unfused")
    p.add_argument("--debug", action="store_true", help="also check internals with step_debug*.onnx")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-steps", type=int, help="only the first steps of each chunk (quick runs)")
    p.add_argument("--quiet", action="store_true", help="print failures and the summary only")
    args = p.parse_args()
    cases = args.case or sorted(d.name for d in GOLDEN_ROOT.iterdir() if (d / "manifest.json").exists())
    tolerances = load_tolerances()
    failed_total = 0
    for name in cases:
        t0 = time.perf_counter()
        results = check_case(name, args.attention, args.debug, args.threads, tolerances, args.max_steps)
        text, _stage, failed = report(results)
        if args.quiet:
            text = "\n".join(line for line in text.splitlines() if not line.startswith(("ok  ", "INFO")))
        print(f"== {name} ({args.attention}, {time.perf_counter() - t0:.0f} s)\n{text}", flush=True)
        failed_total += failed
    sys.exit(1 if failed_total else 0)


if __name__ == "__main__":
    main()
