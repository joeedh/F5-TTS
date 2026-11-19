# DiT step graph on WebGPU: attention form, sequence length, graph capture (plan 01)

Measured 2026-11-19 on the laptop ([webgpu-laptop.md](webgpu-laptop.md)). Plan: [../plans/webgpu-01-dit-export.md](../plans/webgpu-01-dit-export.md), decisions D2 and D3, step 4.

**Decisions:**

- D3: `mha` (`com.microsoft.MultiHeadAttention`).
- D2: dynamic n.
- Graph capture: not used. It fails on the real model in ORT Web 1.31.1.

## What was run

- Graphs: the full fp32 step graph of `F5TTS_v1_Base` (batch 2 for CFG, all 22 blocks, conv position embedding, `proj_out` and the ODE update), exported by `tools/webgpu/export_dit.py` with precomputed rotary tables.
  - Dynamic n: `step_mha.onnx` (`--attention mha`).
  - Static n: `step_mha_n1024.onnx` and `step_mha_n2048.onnx` (`--attention mha --static-n N`); `step_n1024.onnx` and `step_n2048.onnx` (unfused, for graph capture).
- Page: `web/bench/dit-step/` (`main.ts`, driven by `run.mjs`). onnxruntime-web 1.31.1, `webgpu` EP, inputs and outputs on GPU buffers, random inputs.
  - Each configuration runs in a fresh page.
  - `first`: the first call at an n, measured to `queue.onSubmittedWorkDone()`. For the dynamic graph, n = 1024 is the cold start; 2048 and then 1913 (a real chunk length) are new n values after warming up at another n.
  - `single`: 3 warm-ups, then the median of 20 runs, each waiting on the GPU.
  - `chain`: 5 chains of 32 steps, x_next copied into x on the GPU between steps (`copyBufferToBuffer`), median chain divided by 32. This is the real sampling loop minus the text graph.
- Browsers: Chrome 158.0.8127.37 on the RTX 5050 and on the Intel iGPU; Edge 157.0.4341.60 on the RTX 5050.
- Raw results: `web/bench/dit-step/results/*.json`.

## D3: attention form

Taken mostly from steps 2 and 3, with step 4's timing:

- `mha` passes the browser parity runner on both GPUs: all 1,228 comparisons over the 6 real cases, worst at 10 percent of tolerance.
- `onnx23` (opset 23 `Attention`) was correct but took about 25 s a step in Chrome, likely because the op runs on the CPU. Excluded.
- Unfused attention needs 6.5 to 7.3 GB of the RTX 5050's 8 GB at n = 1913 (the (2, 16, n, n) scores, about 0.47 GB per block), against 1.7 GB for `mha`. It is also no faster. It stays as the parity baseline only.

Step time, unfused against `mha`, static n (ms per step in a chain):

| Browser, GPU | n | `mha` | unfused |
|---|---|---|---|
| Chrome, NVIDIA | 1024 | 397 | 413 |
| Chrome, NVIDIA | 2048 | 941 | 1025 |
| Chrome, Intel | 1024 | 1263 | 1295 |
| Chrome, Intel | 2048 | 3000 | 3091 |
| Edge, NVIDIA | 1024 | 397 | 414 |
| Edge, NVIDIA | 2048 | 942 | 1024 |

## D2: dynamic against static n

Per-step time of the `mha` graph (ms; chain per step, with the single-run median in brackets):

| Browser, GPU | n | dynamic | static |
|---|---|---|---|
| Chrome, NVIDIA | 1024 | 406 (405) | 397 (398) |
| Chrome, NVIDIA | 2048 | 961 (964) | 941 (945) |
| Chrome, NVIDIA | 1913 | 880 (885) | — |
| Chrome, Intel | 1024 | 1162 (1161) | 1263 (1150) |
| Chrome, Intel | 2048 | 2776 (2811) | 3000 (2653) |
| Chrome, Intel | 1913 | 2660 (2568) | — |
| Edge, NVIDIA | 1024 | 402 (406) | 397 (401) |
| Edge, NVIDIA | 2048 | 960 (964) | 942 (943) |
| Edge, NVIDIA | 1913 | 879 (883) | — |

- Steady state: dynamic is 2 percent slower than static on NVIDIA and no slower on Intel. The Intel numbers vary by up to 10 percent between single runs and chains, which is more than the dynamic/static gap.
- First call at a new n, against that n's steady single-run time:

  | Browser, GPU | 2048 after 1024 | 1913 after 2048 |
  |---|---|---|
  | Chrome, NVIDIA | 971 against 964 | 879 against 885 |
  | Chrome, Intel | 3051 against 2811 | 2531 against 2568 |
  | Edge, NVIDIA | 972 against 964 | 887 against 883 |

  A new n costs nothing measurable on NVIDIA and at most about 0.25 s on Intel. This suggests ORT's shaders for this graph are not specialized on n, so a new n reuses the pipelines; not checked in ORT's source.
- The cold first call (the very first run of a session) is 1.4 s on Chrome NVIDIA, 2.7 s on Edge NVIDIA and 3.3 s on Intel. ORT uploads the weights at the first run, so this is part of model load, not of per-chunk cost.
- Both D2 rules pass: dynamic is under 10 percent slower in steady state, and its new-n latency is under 1 s. Dynamic n is kept; no buckets and no mask path.

## Graph capture

- `enableGraphCapture: true` fails at the first run in ORT Web 1.31.1, in Chrome and Edge, on both GPUs:
  `Failed to execute 'createBindGroup' on 'GPUDevice': ... Failed to read the 'buffer' property from 'GPUBufferBinding': Required member is undefined.`
  - It fails for every `mha` graph (tiny and base), and for the base unfused graph, static n, even at n = 64.
  - It works only for the tiny unfused graph. The failure is not explained; a guess that it came from the rotary-table `Gather` was tested and was wrong.
  - With capture, passing preallocated `fetches` fails earlier, with a `BufferManager::MemCpy` size error, so the bench lets ORT own the outputs.
- What capture could buy: it removes per-run JS and command-encoding overhead, not GPU work. The base step graph at n = 64, where GPU work is tiny, takes 36 ms on Chrome NVIDIA, which bounds that overhead. That is under 10 percent of a step at n = 1024 (about 9 percent) and about 4 percent at 2048, and less on Intel.
- Not worth chasing now. Worth retrying with a later ORT Web release; plan 07 owns it.

## Per-step time and real-time factor

The full step, `mha`, dynamic n, chain time per step. Real-time factor is 32 steps over the new audio a chunk of that n produces (as in the spike's table: 5.7 s at n = 1024 and 16.6 s at 2048). The text graph, the vocoder and the reference-audio prep are not included.

| Browser, GPU | n | ms per step | 32 steps, s | RTF |
|---|---|---|---|---|
| Chrome, NVIDIA | 1024 | 406 | 13.0 | 2.3 |
| Chrome, NVIDIA | 2048 | 961 | 30.8 | 1.9 |
| Edge, NVIDIA | 1024 | 402 | 12.9 | 2.3 |
| Edge, NVIDIA | 2048 | 960 | 30.7 | 1.9 |
| Chrome, Intel | 1024 | 1162 | 37.2 | 6.5 |
| Chrome, Intel | 2048 | 2776 | 88.8 | 5.3 |

- The spike projected 2.48 (Chrome NVIDIA, n = 1024, fp32, unfused blocks plus convpos). The full `mha` step comes in at 2.3, so the parts the projection left out cost less than `mha` saved.
- Generation is 2 to 6.5 times slower than real time. Plan 07 (fp16, fewer steps, faster convpos) is where that changes.

## Caveats

- One run per configuration. Intel times vary by up to 10 percent between single runs and chains; NVIDIA by about 2 percent.
- Power source and thermal state were not recorded.
- Inputs are random. Step time does not depend on the values.
