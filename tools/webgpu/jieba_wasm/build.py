"""Plan 03 step 3 (D1): builds jieba-rs 0.11.0 to wasm, for web/src/text/segment.ts.

  python tools/webgpu/jieba_wasm/build.py

Needs a Rust toolchain with the wasm32-unknown-unknown target (rustup target add wasm32-unknown-unknown).
jieba-rs is pinned to 0.11.0, the version rjieba 0.2.2 embeds (its SBOM), and Cargo.lock pins its
dependencies to the SBOM's versions. The dictionary is jieba-rs's own src/data/dict.txt, embedded
uncompressed: the default-dict feature unpacks it with zstd, which needs a C compiler for wasm.

Writes web/src/text/jieba.wasm and records the inputs and hashes in build.json. Build paths are remapped,
and the build fails if a local path is left in the wasm. Also writes jieba.wasm.LICENSE.txt next to the wasm,
from licenses/: jieba-rs (MIT) and jieba (MIT), the source of the dictionary and HMM model jieba-rs ships.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "web/src/text/jieba.wasm"
NOTICE_OUT = REPO / "web/src/text/jieba.wasm.LICENSE.txt"
NOTICE_HEAD = (
    "jieba.wasm is jieba-rs {version} built for the browser (tools/webgpu/jieba_wasm). It embeds jieba-rs's\n"
    "dictionary (src/data/dict.txt), which is derived from jieba's, and its HMM model. Both projects are MIT.\n"
)
RECORD = HERE / "build.json"
TARGET = "wasm32-unknown-unknown"
JIEBA_RS = "0.11.0"


def run(*cmd: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(cmd, cwd=HERE, env=env, check=True, capture_output=True, text=True).stdout


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    meta = json.loads(run("cargo", "metadata", "--format-version", "1", "--locked"))
    pkg = next(p for p in meta["packages"] if p["name"] == "jieba-rs")
    assert pkg["version"] == JIEBA_RS, pkg["version"]
    dict_txt = Path(pkg["manifest_path"]).parent / "src/data/dict.txt"
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))

    env = dict(os.environ)
    env["JIEBA_DICT"] = str(dict_txt)
    env["RUSTFLAGS"] = f"--remap-path-prefix={cargo_home}=/cargo --remap-path-prefix={HERE}=/build"
    run("cargo", "build", "--release", "--locked", "--target", TARGET, env=env)
    built = HERE / f"target/{TARGET}/release/jieba_wasm_f5.wasm"

    data = built.read_bytes()
    leaks = {m.decode() for m in re.findall(rb"[A-Za-z]:[\\/][\x20-\x7e]{0,40}", data)}
    leaks |= {s for s in (str(Path.home()), Path.home().name) if s.encode() in data}
    assert not leaks, f"local paths left in the wasm: {sorted(leaks)}"

    shutil.copyfile(built, OUT)
    parts = [NOTICE_HEAD.format(version=JIEBA_RS)]
    for name, title in [
        ("jieba-rs.txt", "jieba-rs (github.com/messense/jieba-rs)"),
        ("jieba.txt", "jieba (github.com/fxsjy/jieba)"),
    ]:
        parts.append(f"\n==== {title} ====\n\n" + (HERE / "licenses" / name).read_text(encoding="utf-8"))
    NOTICE_OUT.write_text("".join(parts), encoding="utf-8", newline="\n")
    record = {
        "jieba-rs": JIEBA_RS,
        "rustc": run("rustc", "--version").strip(),
        "dict.txt sha256": sha256(dict_txt),
        "jieba.wasm sha256": sha256(OUT),
        "jieba.wasm bytes": OUT.stat().st_size,
    }
    RECORD.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
