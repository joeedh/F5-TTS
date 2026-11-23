"""Plan 03 step 1: what upstream's text front end makes of each input, for the TypeScript tests.

  .venv313\\Scripts\\python.exe tools/webgpu/text_reference.py
  .venv313\\Scripts\\python.exe tools/webgpu/text_reference.py --input corpus.txt --out corpus.json
  python tools/webgpu/text_reference.py --check     (CI: the committed edge-case dump still matches Python)

Reads one input per line (lines starting with # and empty lines are skipped; \\u{XXXX} stands for that
code point) and writes, per line:
  segments      rjieba.cut of the text after convert_char_to_pinyin's translate
  pinyin_calls  every lazy_pinyin call convert_char_to_pinyin makes: [input, output]
  tokens, ids   convert_char_to_pinyin's tokens and their vocab ids (unknown is 0)
  fixed_ref     the reference-text fix-up of preprocess_ref_audio_text
  spaced_ref    then infer_batch_process's trailing-space rule
  utf8_len      len(text.encode("utf-8"))
  chunks        chunk_text(text, m) for each m in CHUNK_MAX_CHARS
It also reproduces every golden case's max_chars, chunks, ref_text_spaced, joined_text, tokens and ids
from its manifest, and exits 1 on any difference. Run upstream's own functions; the two rules that sit
inside larger functions are copied and asserted against their source (COPIED_LINES).
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import re
import sys
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

import f5_tts.infer.utils_infer as ui
import f5_tts.model.utils as mu


REPO = Path(__file__).resolve().parents[2]
GOLDEN_ROOT = REPO / "testdata/webgpu"
EDGE_CASES = GOLDEN_ROOT / "text/edge_cases.txt"
VOCAB = REPO / "src/f5_tts/infer/examples/vocab.txt"
CHUNK_MAX_CHARS = [10, 25, 50, 135]

# copied below; asserted to still be in upstream's source
COPIED_LINES = {
    "preprocess_ref_audio_text": [
        'if not ref_text.endswith(". ") and not ref_text.endswith("。"):',
        'if ref_text.endswith("."):',
        'ref_text += " "',
        'ref_text += ". "',
    ],
    "infer_batch_process": ['if len(ref_text[-1].encode("utf-8")) == 1:', 'ref_text = ref_text + " "'],
    "infer_process": [
        'max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * speed)'
    ],
}


def check_copied() -> None:
    for fn, lines in COPIED_LINES.items():
        src = inspect.getsource(getattr(ui, fn))
        for line in lines:
            assert line in src, f"{fn} changed; update text_reference.py: {line}"


def fix_ref_text(ref_text: str) -> str:
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "
    return ref_text


def add_trailing_space(ref_text: str) -> str:
    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "
    return ref_text


def max_chars(ref_text: str, samples: int, sr: int, speed: float) -> int:
    return int(len(ref_text.encode("utf-8")) / (samples / sr) * (22 - samples / sr) * speed)


def decode_line(line: str) -> str:
    return re.sub(r"\\u\{([0-9a-fA-F]+)\}", lambda m: chr(int(m.group(1), 16)), line)


def read_inputs(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [decode_line(x) for x in lines if x and not x.startswith("#")]


class Recorder:
    def __init__(self) -> None:
        self.segments: list[str] = []
        self.pinyin_calls: list[list[Any]] = []  # Any: [input str, output list[str]]


@contextlib.contextmanager
def recording() -> Iterator[Recorder]:
    """Wraps the rjieba.cut and lazy_pinyin that convert_char_to_pinyin calls, to record their results."""
    rec = Recorder()
    real_rjieba, real_lazy = mu.rjieba, mu.lazy_pinyin

    class Jieba:
        @staticmethod
        def cut(text: str) -> list[str]:
            segs = list(real_rjieba.cut(text))
            rec.segments.extend(segs)
            return segs

    def lazy(text: str, **kwargs: Any) -> list[str]:  # Any: pypinyin's keyword arguments
        out = real_lazy(text, **kwargs)
        # the all-3-byte branch indexes the output per character, which assumes one item per character
        if len(text.encode("utf-8")) == 3 * len(text):
            assert len(out) == len(text), f"lazy_pinyin({text!r}) gave {len(out)} items for {len(text)} characters"
        rec.pinyin_calls.append([text, out])
        return out

    mu.rjieba, mu.lazy_pinyin = Jieba, lazy
    try:
        yield rec
    finally:
        mu.rjieba, mu.lazy_pinyin = real_rjieba, real_lazy


def tokens_of(text: str) -> tuple[list[str], list[str], list[list[Any]]]:  # Any: see Recorder
    with recording() as rec:
        tokens = mu.convert_char_to_pinyin([text])[0]
    return tokens, rec.segments, rec.pinyin_calls


def dump_line(text: str, vocab: dict[str, int]) -> dict[str, Any]:  # Any: json
    tokens, segments, calls = tokens_of(text)
    fixed = fix_ref_text(text)
    return {
        "text": text,
        "segments": segments,
        "pinyin_calls": calls,
        "tokens": tokens,
        "ids": [vocab.get(t, 0) for t in tokens],
        "fixed_ref": fixed,
        "spaced_ref": add_trailing_space(fixed),
        "utf8_len": len(text.encode("utf-8")),
        "chunks": {str(m): ui.chunk_text(text, max_chars=m) for m in CHUNK_MAX_CHARS},
    }


def check_golden(name: str, vocab: dict[str, int]) -> list[str]:
    """Reproduces a golden case's text stage and tokens; returns the differences."""
    folder = GOLDEN_ROOT / name
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    def same(what: str, got: Any, want: Any) -> None:  # Any: json values
        if got != want:
            errors.append(f"{name} {what}: got {got!r}, golden {want!r}")

    ref_text = fix_ref_text(man["input"]["ref_text"])
    same("prep.ref_text", ref_text, man["prep"]["ref_text"])
    prep = next(t for t in man["tensors"] if t["name"] == "prep.audio")
    mc = max_chars(ref_text, prep["shape"][-1], man["prep"]["sample_rate"], man["settings"]["speed"])
    same("max_chars", mc, man["text"]["max_chars"])
    chunks = ui.chunk_text(man["input"]["gen_text"], max_chars=mc)
    same("chunks", chunks, man["text"]["chunks"])
    spaced = add_trailing_space(ref_text)
    same("ref_text_spaced", spaced, man["text"]["ref_text_spaced"])
    for c, chunk in enumerate(man["chunks"]):
        joined = spaced + chunks[c] if c < len(chunks) else None
        same(f"c{c}.joined_text", joined, chunk["joined_text"])
        tokens, _, _ = tokens_of(chunk["joined_text"])
        same(f"c{c}.tokens", tokens, chunk["tokens"])
        ids = np.load(folder / f"c{c}.dit.text_ids.npy")[0].tolist()
        same(f"c{c}.ids", [vocab.get(t, 0) for t in tokens], ids)
    return errors


def versions() -> dict[str, str]:
    sbom = next((Path(mu.rjieba.__file__).parent.parent / f"rjieba-{version('rjieba')}.dist-info/sboms").glob("*.json"))
    components = json.loads(sbom.read_text(encoding="utf-8")).get("components", [])
    jieba_rs = next((c["version"] for c in components if c.get("name") == "jieba-rs"), "unknown")
    return {"rjieba": version("rjieba"), "jieba-rs": jieba_rs, "pypinyin": version("pypinyin"), "python": sys.version}


def check_dump(dump: dict[str, Any], committed: dict[str, Any]) -> int:  # Any: json
    """Plan 03 R1: a new rjieba or pypinyin shows up as a changed line. Returns the number of differences."""
    ours = {k: v for k, v in dump["versions"].items() if k != "python"}
    theirs = {k: v for k, v in committed["versions"].items() if k != "python"}
    if ours != theirs:
        print(f"versions differ: now {ours}, committed {theirs}")
    diffs = [(a, b) for a, b in zip(dump["lines"], committed["lines"], strict=False) if a != b]
    diffs += [(a, None) for a in dump["lines"][len(committed["lines"]) :]]
    for a, b in diffs[:10]:
        keys = [k for k in a if b is None or a[k] != b.get(k)]
        print(f"differs: {a['text']!r} in {keys}")
    print(f"dump check: {len(diffs)} of {len(dump['lines'])} lines differ from the committed dump")
    return len(diffs) + (ours != theirs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=EDGE_CASES)
    p.add_argument("--out", type=Path, help="default: the input with .json in place of .txt")
    p.add_argument("--no-golden", action="store_true", help="skip reproducing the golden cases")
    p.add_argument("--check", action="store_true", help="compare with the existing --out file instead of writing it")
    args = p.parse_args()
    if isinstance(sys.stdout, io.TextIOWrapper):  # a Windows pipe defaults to cp1252, which has no CJK
        sys.stdout.reconfigure(encoding="utf-8")
    check_copied()
    vocab, _ = mu.get_tokenizer(str(VOCAB), "custom")
    assert isinstance(vocab, dict)
    out: Path = args.out or args.input.with_suffix(".json")
    lines = read_inputs(args.input)
    dump = {
        "versions": versions(),
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "lines": [dump_line(t, vocab) for t in lines],
    }
    failed = 0
    if args.check:
        failed += check_dump(dump, json.loads(out.read_text(encoding="utf-8")))
    else:
        out.write_text(json.dumps(dump, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{len(lines)} lines -> {out}; versions {dump['versions']}")
    if args.no_golden:
        sys.exit(1 if failed else 0)
    for folder in sorted(d for d in GOLDEN_ROOT.iterdir() if (d / "manifest.json").exists()):
        errors = check_golden(folder.name, vocab)
        failed += len(errors)
        print(f"golden {folder.name}: {'ok' if not errors else f'{len(errors)} differences'}")
        for e in errors:
            print(f"  {e}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
