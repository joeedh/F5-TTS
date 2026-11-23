"""Plan 04 step 7 (D5): the Seed-TTS subset for the quality comparison, extracted from the local tar.

  .venv313/Scripts/python.exe -I tools/webgpu/quality_set.py seedtts_testset.tar

Writes data/seedtts_testset/{en,zh}/meta.lst (whole, as eval_seedtts_testset.py expects) and the prompt wavs
of the chosen utterances under prompt-wavs/. The eval scores only utterances whose generated wav exists, so
generating the subset is enough. Picks N utterances per language with a fixed seed, gives each a uint32
generation seed, and writes data/seedtts_testset/quality_subset.json for quality_torch.py and
web/bench/quality/. Output paths are built here from validated names, never taken from the tar as they are.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/seedtts_testset"
N = 100
PICK_SEED = 20261122
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tar", type=Path)
    p.add_argument("-n", type=int, default=N, help="utterances per language")
    args = p.parse_args()
    rng = np.random.default_rng(PICK_SEED)
    subset: dict[str, list[dict[str, str | int]]] = {}
    with tarfile.open(args.tar) as tar:
        for lang in ("en", "zh"):
            meta_bytes = tar.extractfile(f"seedtts_testset/{lang}/meta.lst")
            assert meta_bytes is not None
            meta = meta_bytes.read()
            lines = [ln for ln in meta.decode("utf-8").splitlines() if ln.strip()]
            folder = OUT / lang / "prompt-wavs"
            folder.mkdir(parents=True, exist_ok=True)
            (OUT / lang / "meta.lst").write_bytes(meta)
            picked = sorted(rng.choice(len(lines), size=args.n, replace=False).tolist())
            seeds = rng.integers(0, 2**32, size=args.n, dtype=np.uint64).tolist()
            rows: list[dict[str, str | int]] = []
            for i, seed in zip(picked, seeds, strict=True):
                utt, prompt_text, prompt_wav, target_text = lines[i].split("|")[:4]
                folder_name, wav_name = prompt_wav.split("/")
                assert folder_name == "prompt-wavs" and SAFE.match(wav_name) and SAFE.match(utt), lines[i]
                data = tar.extractfile(f"seedtts_testset/{lang}/prompt-wavs/{wav_name}")
                assert data is not None, wav_name
                (folder / wav_name).write_bytes(data.read())
                rows.append(
                    {
                        "utt": utt,
                        "prompt_text": prompt_text,
                        "prompt_wav": f"{lang}/prompt-wavs/{wav_name}",
                        "target_text": target_text,
                        "seed": int(seed),
                    }
                )
            subset[lang] = rows
            print(f"{lang}: {len(rows)} of {len(lines)} utterances")
    out = OUT / "quality_subset.json"
    out.write_text(json.dumps(subset, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
