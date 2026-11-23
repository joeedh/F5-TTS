"""Plan 04 step 2: torch.randn on the CPU for several seeds and sizes, for web/test/pipeline/noise.test.ts.

  .venv/Scripts/python.exe tools/webgpu/noise_fixtures.py

Each seed draws SIZES in turn from one manual_seed, so the stream's continuation is covered too. Records the
CPU capability: torch's Box-Muller is vectorized only on AVX2, so other machines give slightly other noise.
Writes testdata/webgpu/pipeline/randn_cases.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "testdata/webgpu/pipeline/randn_cases.json"
SEEDS = [0, 1, 12345, 2**31, 2**32 - 1]
SIZES = [16, 17, 31, 100, 1031]  # with the tail redraws, past two refills of the 624-word state


def main() -> None:
    cases = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        draws = [torch.randn(n).tolist() for n in SIZES]
        cases.append({"seed": seed, "sizes": SIZES, "draws": draws})
    torch.manual_seed(2**32 + 5)
    high = torch.randn(16)
    torch.manual_seed(5)
    assert torch.equal(high, torch.randn(16)), "the CPU generator no longer drops the seed's high bits"
    data = {"torch": torch.__version__, "cpu_capability": torch.backends.cpu.get_cpu_capability(), "cases": cases}
    OUT.write_text(json.dumps(data) + "\n", encoding="utf-8")
    print(f"{len(cases)} seeds -> {OUT} ({OUT.stat().st_size} bytes), {data['cpu_capability']}")


if __name__ == "__main__":
    main()
