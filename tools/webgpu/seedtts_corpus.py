"""Plan 03 (D4): the Seed-TTS eval texts as a local test corpus for the text front end. Not committed.

  .venv313\\Scripts\\python.exe tools/webgpu/seedtts_corpus.py path\\to\\seedtts_testset.tar

Reads en/meta.lst, zh/meta.lst and zh/hardcase.lst from the tar without extracting anything else. Each line
is `id|prompt text|prompt wav|target text`. Writes the unique prompt and target texts, in order of first use,
to testdata/webgpu/text/corpus/seedtts_en.txt and seedtts_zh.txt (zh includes the hard cases). Then:

  .venv313\\Scripts\\python.exe tools/webgpu/text_reference.py --no-golden --input testdata/webgpu/text/corpus/seedtts_zh.txt
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "testdata/webgpu/text/corpus"
LISTS = {"en": ["en/meta.lst"], "zh": ["zh/meta.lst", "zh/hardcase.lst"]}


def texts(tar: tarfile.TarFile, member: str) -> list[str]:
    name = next(n for n in tar.getnames() if n.endswith(member))
    f = tar.extractfile(name)
    assert f is not None, name
    out: list[str] = []
    for line in f.read().decode("utf-8").splitlines():
        fields = line.split("|")
        assert len(fields) >= 4, (member, line)
        out += [fields[1], fields[3]]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tar", type=Path)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.tar) as tar:
        for lang, members in LISTS.items():
            seen: dict[str, None] = {}
            for m in members:
                for t in texts(tar, m):
                    seen.setdefault(t.strip(), None)
            lines = [t for t in seen if t]
            # text_reference.py reads # as a comment and \u{...} as an escape; neither may appear literally
            assert not any(t.startswith("#") or "\\u{" in t for t in lines), lang
            (OUT / f"seedtts_{lang}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"{lang}: {len(lines)} unique texts -> {OUT / f'seedtts_{lang}.txt'}")


if __name__ == "__main__":
    main()
