"""Behaviour tripwire for the typing work (docs/plans/python-pyright-tooling.md, step 2).

Run on untouched code with --write to store tools/typing_guard/golden.json, then run
without arguments after every batch. Exit codes: 0 all match, 1 mismatch or error,
2 no mismatch but the numeric part was skipped because the environment differs.

Sections:
    syntax      every .py under src/f5_tts compiles, and parses with the 3.10 grammar
    imports     the allow-listed modules import
    numeric     CFM.forward and CFM.sample on a tiny random DiT, hashed (same torch only)
    helpers     convert_char_to_pinyin, get_epss_timesteps, chunk_text on fixed inputs
    signatures  parameter names, kinds and defaults of public functions and classes

This is a tripwire, not a proof. Annotations are ignored by every comparison.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import platform
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "f5_tts"
GOLDEN = Path(__file__).with_name("golden.json")

# Modules imported and signature-snapshotted. Everything else runs module-level code
# (checkpoint loading, downloads, CLI parsing) and is only syntax-checked.
EXTRA_MODULES = ["f5_tts.api", "f5_tts.device_utils", "f5_tts.infer.utils_infer"]

VOCAB = " abcdefghij.,!"

PINYIN_INPUTS = [
    "Hello, world! This is a test.",
    "你好，世界。今天天气很好。",
    "我 said “hello” to ‘him’; 然后走了.",
    "Mixed 中文 and English 123",
    "",
    "重庆的行长很重要",
    "café naïve résumé",
]

CHUNK_INPUTS = [
    "Hello there. This is a short text, with a comma; and a semicolon! Is it split? Yes, it is.",
    "第一句话。第二句话！第三句话？第四句话，还有更多的内容在这里，直到超过限制为止。",
    "Mixed 中文 and English. 这是一个混合句子! Another one, right?",
    "A single very long sentence without any terminal punctuation that keeps going and going " * 3,
    "",
]


def tensor_record(t: torch.Tensor) -> dict[str, Any]:
    a = t.detach().to(torch.float32).cpu().contiguous()
    assert bool(torch.isfinite(a).all()), "non-finite output"
    assert float(a.abs().max()) > 0, "all-zero output: the guard would not detect anything"
    d = a.to(torch.float64)
    return {
        "shape": list(a.shape),
        "sha256": hashlib.sha256(a.numpy().tobytes()).hexdigest(),
        "mean": round(float(d.mean()), 8),
        "std": round(float(d.std(correction=0)), 8),
        "absmax": round(float(d.abs().max()), 8),
    }


def reseed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def environment() -> dict[str, str]:
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
    }
    for pkg in ["torchdiffeq", "x_transformers", "numpy", "rjieba", "pypinyin"]:
        try:
            env[pkg] = importlib.metadata.version(pkg.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            env[pkg] = "not installed"
    return env


def source_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def check_syntax() -> list[str]:
    errors = []
    for path in source_files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
            compile(text, rel, "exec", dont_inherit=True)
            ast.parse(text, rel, feature_version=(3, 10))
        except (SyntaxError, ValueError) as e:
            errors.append(f"{rel}: {type(e).__name__}: {e}")
    return errors


def module_names() -> list[str]:
    names = list(EXTRA_MODULES)
    for path in sorted((SRC / "model").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = path.relative_to(SRC.parent).with_suffix("").parts
        names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return names


def check_imports() -> dict[str, str]:
    failed = {}
    for name in module_names():
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - report any import failure, whatever its type
            failed[name] = f"{type(e).__name__}: {e}"
    return failed


def build_cfm() -> Any:
    from f5_tts.model import CFM, DiT

    reseed(0)
    vocab_char_map = {c: i for i, c in enumerate(VOCAB)}
    dit = DiT(
        dim=64,
        depth=2,
        heads=2,
        dim_head=32,
        ff_mult=2,
        mel_dim=100,
        text_num_embeds=len(vocab_char_map),
        text_dim=32,
        conv_layers=2,
    )
    # DiT zero-initialises its output layers, so a fresh model returns all zeros.
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for p in dit.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.05)
    cfm = CFM(transformer=dit, vocab_char_map=vocab_char_map)
    cfm.eval()
    return cfm


def numeric_section() -> dict[str, Any]:
    torch.set_num_threads(1)
    cfm = build_cfm()
    gen = torch.Generator().manual_seed(99)
    out: dict[str, Any] = {}

    mel = torch.randn(2, 24, 100, generator=gen)
    reseed(1)
    loss, cond, pred = cfm(mel, ["abc de", "fghij ."], lens=torch.tensor([24, 20]))
    out["forward"] = {"loss": tensor_record(loss), "cond": tensor_record(cond), "pred": tensor_record(pred)}

    single = torch.randn(1, 16, 100, generator=gen)
    pair = torch.randn(2, 16, 100, generator=gen)
    cases = {
        # single item, EPSS table path, classifier-free guidance, sway sampling
        "sample_single_epss": dict(cond=single, text=["abc de"], duration=32, steps=5, cfg_strength=2.0, sway=-1.0),
        # batch of two with a mask and different prompt lengths, linspace path
        "sample_batch_mask": dict(
            cond=pair, text=["abc de", "fghij ."], duration=30, steps=4, cfg_strength=2.0, lens=torch.tensor([16, 12])
        ),
        # no classifier-free guidance
        "sample_no_cfg": dict(cond=single, text=["abc de"], duration=24, steps=4, cfg_strength=0.0),
    }
    for name, case in cases.items():
        reseed(2)
        sampled, trajectory = cfm.sample(
            case["cond"],
            case["text"],
            case["duration"],
            lens=case.get("lens"),
            steps=case["steps"],
            cfg_strength=case["cfg_strength"],
            sway_sampling_coef=case.get("sway"),
            seed=7,
        )
        out[name] = {"out": tensor_record(sampled), "trajectory": tensor_record(trajectory)}
    return out


def helpers_section() -> dict[str, Any]:
    from f5_tts.infer.utils_infer import chunk_text
    from f5_tts.model.utils import convert_char_to_pinyin, get_epss_timesteps

    steps = [3, 4, 5, 6, 7, 8, 10, 12, 16, 20]
    return {
        "pinyin": convert_char_to_pinyin(PINYIN_INPUTS),
        "pinyin_no_polyphone": convert_char_to_pinyin(PINYIN_INPUTS, polyphone=False),
        "epss": {str(n): get_epss_timesteps(n, device="cpu", dtype=torch.float32).tolist() for n in steps},
        "chunks_default": [chunk_text(t) for t in CHUNK_INPUTS],
        "chunks_40": [chunk_text(t, max_chars=40) for t in CHUNK_INPUTS],
    }


def describe(fn: Any) -> list[list[str]]:
    rows = []
    for p in inspect.signature(fn).parameters.values():
        default = "<required>" if p.default is p.empty else re.sub(r"0x[0-9a-fA-F]+", "0x?", repr(p.default))
        rows.append([p.name, p.kind.name, default])
    return rows


def signatures_section() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for modname in module_names():
        module = sys.modules[modname]
        for name, obj in sorted(vars(module).items()):
            if name.startswith("_") or getattr(obj, "__module__", None) != modname:
                continue
            if inspect.isfunction(obj):
                snapshot[f"{modname}.{name}"] = describe(obj)
            elif inspect.isclass(obj):
                for attr, member in sorted(vars(obj).items()):
                    if attr.startswith("_") and attr != "__init__":
                        continue
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    elif isinstance(member, property):
                        member = member.fget
                    if inspect.isfunction(member) or hasattr(member, "__wrapped__"):
                        snapshot[f"{modname}.{name}.{attr}"] = describe(member)
    return snapshot


def collect(with_numeric: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"environment": environment()}
    result["syntax_errors"] = check_syntax()
    result["import_failures"] = check_imports()
    if result["import_failures"]:
        return result
    if with_numeric:
        result["numeric"] = numeric_section()
    result["helpers"] = helpers_section()
    result["signatures"] = signatures_section()
    return json.loads(json.dumps(result))  # normalise tuples and ints the way the stored file is


def diff(old: Any, new: Any, path: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        lines = []
        for key in sorted(set(old) | set(new)):
            if key not in old:
                lines.append(f"  added   {path}/{key}")
            elif key not in new:
                lines.append(f"  removed {path}/{key}")
            else:
                lines.extend(diff(old[key], new[key], f"{path}/{key}"))
        return lines
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [line for i, (o, n) in enumerate(zip(old, new)) for line in diff(o, n, f"{path}[{i}]")]
    return [] if old == new else [f"  changed {path}: {str(old)[:80]} -> {str(new)[:80]}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="store the current results as golden.json")
    args = parser.parse_args()

    stored_env = None if args.write else json.loads(GOLDEN.read_text(encoding="utf-8"))["environment"]
    cur_env = environment()
    # Numerics are only comparable within one torch build, so skip them elsewhere.
    run_numeric = stored_env is None or all(stored_env[k] == cur_env[k] for k in ["torch", "system", "machine"])
    result = collect(with_numeric=run_numeric)
    problems = [f"syntax: {e}" for e in result["syntax_errors"]]
    problems += [f"import {m}: {e}" for m, e in result["import_failures"].items()]

    if args.write:
        if problems:
            print("\n".join(problems))
            print("Not writing golden.json: fix these on the baseline first.")
            return 1
        keep = {k: result[k] for k in ["environment", "numeric", "helpers", "signatures"]}
        GOLDEN.write_text(json.dumps(keep, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(f"Wrote {GOLDEN.relative_to(ROOT)} (torch {result['environment']['torch']})")
        return 0

    stored = json.loads(GOLDEN.read_text(encoding="utf-8"))
    print(f"{'syntax':10} {'FAIL' if result['syntax_errors'] else 'ok'}")
    print(f"{'imports':10} {'FAIL' if result['import_failures'] else 'ok'}")
    for section in ["helpers", "signatures", "numeric"]:
        if section not in result:
            continue  # imports failed, or numeric skipped below
        lines = diff(stored[section], result[section])
        print(f"{section:10} {'FAIL' if lines else 'ok'}")
        if lines:
            problems += [f"{section}:", *lines[:40]]
    if problems:
        print("\n".join(problems))
        return 1
    if not run_numeric:
        print("numeric    SKIPPED: environment differs from golden.json; regenerate it on the baseline commit")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
