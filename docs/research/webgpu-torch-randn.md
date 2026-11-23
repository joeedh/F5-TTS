# torch.randn on the CPU, ported to TypeScript

Plan 04 step 2 (D1). The port is `web/src/pipeline/noise.ts`. Written 2026-11-22, against torch 2.15.0.

## Why

- `CFM.sample` draws its starting noise with `torch.randn(duration, 100)` from torch's global generator (`cfm.py:212`).
- The golden data was made on the CPU, from `manual_seed(0)`, one chunk after another.
- If the port draws the same noise from the same seed, two things follow:
  - the whole pipeline, noise included, can be checked against the golden data;
  - the port and PyTorch can be compared utterance by utterance in plan 04's quality check (D5), since both start from identical noise.

## Sources

All of it is in headers that ship with the torch wheel (`torch/include/`):

| Part | File |
|---|---|
| The generator | `ATen/core/MT19937RNGEngine.h` |
| Uniform floats | `ATen/core/TransformationHelper.h` (`uniform_real`), `ATen/core/DistributionsHelper.h` (`uniform_real_distribution`) |
| The normal fill | `ATen/native/cpu/DistributionTemplates.h` (`normal_kernel`, `normal_fill`, `NormalFill16`) |
| AVX2 `log` and `sincos` | `ATen/native/cpu/avx_mathfun.h` (`log256_ps`, `sincos256_ps`, Cephes polynomials) |

## What torch does

- **Seeding.** `mt19937_engine::init_with_uint32` keeps only `seed & 0xffffffff`, then the standard Mersenne Twister initialisation (`1812433253 * (s ^ (s >> 30)) + j`). So `manual_seed(2**32 + 5)` equals `manual_seed(5)`. `noise_fixtures.py` asserts this.
- **Generator.** The standard 32-bit mt19937: a 624-word state, refilled when exhausted, with the standard tempering.
- **Uniforms.** `uniform_real_distribution<float>(0, 1)` takes one 32-bit output, keeps its low 24 bits, and multiplies by 2^-24. The result is exact in float32 and lies in [0, 1).
- **Normal fill.** For a contiguous float32 tensor of 16 or more elements (`normal_fill`):
  1. Draw all `size` uniforms into the output, in order.
  2. Apply Box-Muller in place, 16 at a time: element `j` and `j + 8` of each block of 16 form a pair.
  3. If `size` is not a multiple of 16, draw 16 fresh uniforms into the last 16 elements (overlapping the last full block) and apply Box-Muller to them again. Almost every `duration * 100` is not a multiple of 16, so this redraw nearly always happens, and it moves the stream on by 16.
- **Box-Muller.** `u1 = 1 - u` (so the `log` never sees 0), `radius = sqrt(-2 log u1)`, `theta = 2 pi u2`; the outputs are `radius * cos(theta)` and `radius * sin(theta)`.
  - On an AVX2 build running on an AVX2 CPU, this is `NormalFill16<float, true>`: float32 throughout, with Cephes `log256_ps` and `sincos256_ps`, and one fused multiply-add for `n * std + mean`.
  - Otherwise it is the scalar version, with `std::log`, `std::cos` and `std::sin`. Its results differ from the AVX2 ones by about 2e-6 (plan 04's pressure test, with `ATEN_CPU_CAPABILITY=default`).
- **Sizes under 16** go through another path (`normal_distribution<double>` per element). It is not ported; the pipeline never draws fewer than 1,600 values.

## The port

- `TorchGenerator(seed)` takes a uint32 seed and has `random()`, `uniform()` and `randn(size)`.
- The AVX2 `log256_ps` and `sincos256_ps` are written out lane by lane, with every float32 operation rounded with `Math.fround`, and the Cephes constants rounded to float32 as `_PS256_CONST` does.
  - With mean 0 and std 1, the fused multiply-add is `n * 1 + 0`, which is exact, so it needs no emulation.
  - `_mm256_sqrt_ps` is correctly rounded, so `fround(Math.sqrt(x))` is the same value.

## Results

- **Bit-exact.** Every chunk of all 7 golden cases matches `c*.noise.y0` exactly, drawn in order from seed 0. That is 0 differing values out of 1,563,200.
- Seeds 0, 1, 12345, 2^31 and 2^32 - 1, each drawing sizes 16, 17, 31, 100 and 1031 in turn, match torch exactly (`testdata/webgpu/pipeline/randn_cases.json`, from `tools/webgpu/noise_fixtures.py`).
- **Mutation checks.** Each of these fails every golden case:
  - `Math.log` in place of `log256_ps`;
  - `Math.sin` and `Math.cos` in place of `sincos256_ps`;
  - dropping the tail redraw.
- Speed: about 9 ms in Node for one 1913-frame chunk (191,300 values), after the first call.

## Limits

- **The noise depends on the CPU of the Python side.** It matches torch on an AVX2 machine only. On a machine without AVX2, or with `ATEN_CPU_CAPABILITY=default`, torch's noise is about 2e-6 away, so the same seed no longer gives bit-identical noise.
  - `make_golden.py` now records `cpu_capability` in the manifest. The existing cases were made on this machine, which reports AVX2.
  - `noise_fixtures.py` records it too, and the test checks that it says AVX2.
  - An AVX-512 CPU does not match either. torch dispatches to its AVX-512 build of this kernel there, and only the AVX2 build defines `CPU_CAPABILITY_AVX2`, so the AVX-512 build takes the scalar Box-Muller. Not tested on such a machine; read from the headers.
- **CUDA noise is unrelated.** CUDA's `randn` uses Philox, so the same seed gives different noise there. D5 draws the PyTorch side's noise on the CPU for that reason.
