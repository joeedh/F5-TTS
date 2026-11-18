# Runtime spike: ORT Web on WebGPU (plan 00, step 3)

Measured 2026-11-18 on the laptop ([webgpu-laptop.md](webgpu-laptop.md)). Plan: [../plans/webgpu-00-scope-and-parity.md](../plans/webgpu-00-scope-and-parity.md), decision D3.

**Decision: ORT Web.** All five D3 criteria pass on the RTX 5050. The hand-written WGSL matmul is no faster than ORT's MatMul. Details and caveats below.

## What was run

- Exporter: `tools/webgpu/spike_export.py` (run in `.venv313`). torch 2.15.0, `torch.onnx.export(dynamo=True)`, onnx 1.23.2, onnxscript 0.7.2. fp16 copies were made with `onnxconverter-common` `convert_float_to_float16(keep_io_types=True)`.
- Models, each with a static shape at n = 256, 1024 and 2048, all with real `F5TTS_v1_Base` weights in fp32:
  - `block`: DiT block 0. Inputs are x (2, n, 1024), the time embedding (2, 1024) and the rotary freqs (1, n, 64).
  - `matmul`: (2n, 1024) x (1024, 2048), using block 0's feed-forward up-projection.
  - `convpos`: `InputEmbedding.conv_pos_embed`, batch 2.
  - `convnext`: `TextEmbedding.text_blocks[0]`, batch 2.
- Inputs are real activations. One CFG call of the full DiT ran at t = 0 with noise x, the mel of `basic_ref_en.wav` as cond (first n/2 frames at most) and random token ids. Hooks recorded what reached each module. Reference outputs are PyTorch fp32 on the CPU.
- Browser side: `web/spikes/ort-block/`, a Vite page plus a Playwright driver (`run.mjs`) and `summarize.mjs`.
  - Runtime: onnxruntime-web 1.31.1, `webgpu` execution provider. Inputs and outputs stay on the GPU (`preferredOutputLocation: "gpu-buffer"`).
  - Timing: 3 warm-up runs, then the median of 20 single runs. Each run waits for `queue.onSubmittedWorkDone()`.
  - Criterion 3: the block runs 22 times in a chain, each output feeding the next input, 5 chains, median taken.
  - Raw results: `web/spikes/ort-block/results/*.json`.
- Browsers: Chrome 157.0.8103.79 and Edge 157.0.4341.60, headless. Both updated themselves after step 2, which recorded 156.
  - NVIDIA runs used `--force_high_performance_gpu`; Intel runs used no flag.
  - The adapter that came back was recorded in every run, and it matched the one intended.

## Export

- Opset 20 by default. The block exports to 80 nodes. Ops: Add, Concat, Cos, Gelu, Gemm, LayerNormalization, MatMul (8), Mul, Neg, Reshape, Sigmoid, Sin, Slice, Softmax, Split, Squeeze, Transpose, Unsqueeze.
- **Attention is not fused.** `scaled_dot_product_attention` became MatMul, Softmax and MatMul, so the full (2, 16, n, n) score matrix is materialised. That is 128 MiB in fp32 at n = 1024 and 512 MiB at n = 2048. See "Attention cost" below.
- `nn.Mish` exports as the opset-18 `Mish` op, and `GELU(approximate="tanh")` as the opset-20 `Gelu` op. ORT's WebGPU EP runs both, so the Mish rewrite in the community export (DakeQQ/F5-TTS-ONNX) is not needed here.
- Python check in ORT CPU 1.31.1 against PyTorch:
  - fp32: relative L2 at most 6e-7 on every model.
  - fp16 copies: 1.5e-4 to 2.3e-4, except the block, whose error grows with n (2.1e-4 at 256, 4.3e-4 at 1024, 1.7e-3 at 2048).
- The rotary freqs are a graph input, as the plan asked. The exporter traced through `x_transformers.apply_rotary_pos_emb` (`rearrange`, `rotate_half`) without trouble.
- Community export notes, read but not used: DakeQQ uses the legacy TorchScript exporter at opset 20 with a hand-written attention. For fp16 it scales Q and K down and computes the softmax in fp32 to avoid overflow. It fuses QKV and the AdaLN linears, and passes RoPE cos/sin tables as inputs.

## Results against the D3 criteria

| # | Criterion | Result | Pass |
|---|---|---|---|
| 1 | Placement | All nodes on `WebGpuExecutionProvider` for every model, fp32 and fp16, on both GPUs and both browsers (log line `All nodes placed on [WebGpuExecutionProvider]`). `MemcpyTransformer modified: 0` everywhere, so no copy nodes were inserted. | yes |
| 2 | fp32 correctness (relative L2 ≤ 1e-4) | Block 5e-7 to 8e-7; matmul 5.5e-7; convpos 9e-7; convnext up to 6.4e-6 | yes |
| 3 | Speed (projected RTF ≤ 5 at n = 1024, fp32, RTX 5050, Chrome) | **2.31**. A 22-block chain takes 413 ms, so 32 steps take 13.2 s for 5.7 s of new audio. Edge: 2.74. | yes |
| 4 | fp16 runs (`shader-f16`) | Runs on both GPUs with no NaN or Inf, and is faster at n ≥ 1024. Accuracy is much worse than expected; see below. | yes (runs) |
| 5 | Integrated GPU | Intel Xe3-LPG gives correct output on every model (block relative L2 4.4e-7). Times are recorded below; they are not judged. | yes |

Hand-written WGSL matmul (`matmul.ts`: a 64x64 tile per workgroup, 4x4 outputs per thread, K tiles of 16, fp32):

| n | NVIDIA ORT / WGSL ms | Intel ORT / WGSL ms |
|---|---|---|
| 256 | 3.3 / 2.6 | 5.0 / 3.4 |
| 1024 | 4.3 / 4.3 (2.0 TFLOPS each) | 11.5 / 9.2 |
| 2048 | 7.0 / 7.6 (2.5 / 2.3 TFLOPS) | 19.3 / 16.8 |

- ORT's MatMul is within 0.8x to 1.5x of the hand-written kernel, and faster at n = 2048 on NVIDIA. The D3 rule "more than 2 times slower" is not triggered.
- For scale, cuBLAS does 7.6 TFLOPS fp32 on this GPU, so WebGPU reaches about a third of that. A WGSL kernel better than this one (vec4 loads, larger tiles, subgroups) could close some of that gap. That belongs to plan 07, not D3.
- Cost of writing the kernel: about 70 lines and one attempt. It was correct the first time (relative L2 5.4e-7).

### Projected real-time factor, 32 steps

"Blocks" is the 22-block chain times 32. "+ convpos" adds one `conv_pos_embed` run per step, which also happens every step. Neither includes the input projection, `norm_out`/`proj_out`, the ODE update or the vocoder.

| Config | n | new audio s | fp32 blocks | fp32 + convpos | fp16 blocks | fp16 + convpos |
|---|---|---|---|---|---|---|
| Chrome NVIDIA | 1024 | 5.7 | 2.31 | 2.48 | 1.71 | 1.84 |
| Chrome NVIDIA | 2048 | 16.6 | 2.33 | 2.45 | 1.47 | 1.55 |
| Edge NVIDIA | 1024 | 5.7 | 2.74 | 2.95 | 2.21 | 2.36 |
| Edge NVIDIA | 2048 | 16.6 | 3.05 | 3.17 | 2.05 | 2.13 |
| Chrome Intel | 1024 | 5.7 | 7.27 | 7.85 | 4.22 | 4.65 |
| Chrome Intel | 2048 | 16.6 | 11.71 | 12.14 | 6.51 | 6.77 |
| Edge Intel | 1024 | 5.7 | 7.52 | 8.18 | 4.60 | 5.02 |
| Edge Intel | 2048 | 16.6 | 22.70 | 23.10 | 11.39 | 11.66 |

- RTF 2.3 means about 13 s to generate one 5.7 s chunk on the discrete GPU in fp32. That is usable but slow, as D3 expected. On the integrated GPU, which is the default on Windows laptops (R10), fp16 is needed just to reach the RTF 5 bar at n = 1024.
- Chaining helps a little. On NVIDIA at n = 1024, the 22-block chain takes 413 ms, against 22 x 22.8 = 502 ms for single runs that each wait on the GPU.

## Findings for later plans

- **fp16 accuracy on the WebGPU EP (plan 05).** The same fp16 models give much larger errors on WebGPU than on ORT CPU:

  | Model | ORT CPU fp16 | ORT WebGPU fp16 (both GPUs) |
  |---|---|---|
  | matmul alone (K = 1024) | 1.5e-4 | 4.3e-3 |
  | block, n = 1024 | 4.3e-4 | 8.2e-3 |
  | convpos | 2.3e-4 | 1.0e-2 |
  | convnext | 2.2e-4 | 3.7e-3 |

  - Rounding the inputs and weights to fp16, with fp32 accumulation, explains the CPU numbers. CUDA fp16 matmul gave 4e-4 in `docs/debugging.md`.
  - The WebGPU matmul error is 30 times larger, which fits accumulation in fp16 over K = 1024. This is an inference from the numbers; the ORT shader source was not checked.
  - The block's maximum absolute error reaches about 1 (output abs max 52).
  - Plan 05 has to measure what this does to audio. The fixes to try are fp32 accumulation in a custom kernel, or keeping MatMul and Conv in fp32 and the rest in fp16.
  - The fp16 gain is modest on NVIDIA: 1.35x on the block at n = 1024 and 1.6x at 2048. It is 1.7x to 1.8x on Intel. The lone MatMul barely speeds up on NVIDIA (4.3 to 4.1 ms).
- **Attention cost at long chunks (plan 01).**
  - From n = 1024 to n = 2048, the block's time grows 3.3x on NVIDIA, 6.2x on Intel in Chrome and 8.9x on Intel in Edge. The MatMul alone grows only 1.4x to 1.8x.
  - The unfused n x n score matrix (512 MiB at n = 2048) is the likely cause, especially on the integrated GPU.
  - Plan 01 should try a fused attention op that the WebGPU EP supports, such as `com.microsoft` `MultiHeadAttention` or the opset-23 `Attention`. It should also measure whether shorter chunks are cheaper overall.
- **`conv_pos_embed` costs about 7 percent of a step** (31 ms on NVIDIA and 103 ms on Intel at n = 1024). The grouped conv runs at only 140 to 560 GFLOPS, so it is a candidate for a custom kernel in plan 07.
- **Edge was slower than Chrome on the same GPU:** 19 percent on NVIDIA at n = 1024, and almost 2x on Intel at n = 2048. Plan 07 should record the browser as well as the GPU.
- **ORT Web device handling (plan 06).** Setting `ort.env.webgpu.device` before the first session has no effect in 1.31.1; ORT creates its own device. ORT also destroys that device when the last session is released. Details and symptoms are in `docs/debugging.md`, 2026-11-18. Plan 06 should keep one session alive for the page's lifetime and use `await ort.env.webgpu.device` for any buffers it shares with ORT.

## Caveats

- One run per configuration: a median of 20 runs, or of 5 chains. Run-to-run variation was not measured this time; step 2 saw about 20 percent on the naive matmul. Power source and thermal state were not recorded.
- The RTF projection covers the block stack and convpos only, so the real figure will be somewhat higher. Plan 04 measures the whole pipeline.
- Static shapes only. Dynamic sequence length could change which kernels ORT picks; that is plan 01's question.
- The spike is frozen. Do not extend it; plan 01 owns the real export.

## Rerunning

```
.venv313\Scripts\python.exe tools/webgpu/spike_export.py       # about 3 minutes; writes web/spikes/ort-block/public/data (gitignored, ~600 MB)
cd web
node spikes/ort-block/run.mjs --channel chrome --gpu nvidia     # also: --gpu intel, --channel msedge, --models a,b, --no-fp16
node spikes/ort-block/summarize.mjs                             # markdown tables from results/*.json
```

`run.mjs` also writes `results/<tag>.log`, the full ORT verbose log (about 7 MB, gitignored), which holds the placement lines.
