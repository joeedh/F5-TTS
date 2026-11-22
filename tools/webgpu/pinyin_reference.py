"""Plan 03 step 4: lazy_pinyin as convert_char_to_pinyin calls it, on every dictionary entry and on random
mixes, for web/test/text/pinyin.test.ts. The output is large and stays local (git ignores corpus/).

  .venv313\\Scripts\\python.exe tools/webgpu/pinyin_reference.py

Writes testdata/webgpu/text/corpus/pinyin_cases.json: {"versions": ..., "cases": [[input, output], ...]}.
Inputs: every PINYIN_DICT character, every PHRASES_DICT word, and RANDOM_CASES random strings joining
phrases, characters, 一 and 不, third-tone characters and non-Han pieces. Seeded, so the file is stable.
"""

from __future__ import annotations

import json
import random
from importlib.metadata import version
from pathlib import Path

from pypinyin import Style, lazy_pinyin
from pypinyin.constants import PHRASES_DICT, PINYIN_DICT


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "testdata/webgpu/text/corpus/pinyin_cases.json"
RANDOM_CASES = 30000
# non-Han pieces convert_char_to_pinyin can pass: CJK punctuation, kana, bopomofo, 〇 (Han to pypinyin),
# a compatibility ideograph, an astral ideograph, fullwidth digits, and characters outside every dictionary
NON_HAN = ["，", "。", "、", "「", "」", "あ", "ア", "ㄅ", "〇", "豈", "𠮶", "１２", "★", "㈠", "䷀", "", "\U0002a6d6"]


def tone3(text: str) -> list[str]:
    return lazy_pinyin(text, style=Style.TONE3, tone_sandhi=True)


def random_cases(rng: random.Random) -> list[str]:
    phrases = sorted(PHRASES_DICT)
    chars = [chr(c) for c in sorted(PINYIN_DICT)]
    third = [c for c in chars if "3" in tone3(c)[0]][:2000]
    out: list[str] = []
    for _ in range(RANDOM_CASES):
        parts: list[str] = []
        for _ in range(rng.randint(1, 5)):
            kind = rng.random()
            if kind < 0.35:
                parts.append(rng.choice(phrases))
            elif kind < 0.55:
                parts.append(rng.choice(chars))
            elif kind < 0.7:
                parts.append(rng.choice("一不"))
            elif kind < 0.85:
                parts.append(rng.choice(third))
            else:
                parts.append(rng.choice(NON_HAN))
        out.append("".join(parts))
    return out


def main() -> None:
    inputs = [chr(c) for c in sorted(PINYIN_DICT)] + sorted(PHRASES_DICT) + random_cases(random.Random(20261122))
    cases = [[t, tone3(t)] for t in inputs]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {"versions": {"pypinyin": version("pypinyin")}, "cases": cases}
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{len(cases)} cases -> {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
