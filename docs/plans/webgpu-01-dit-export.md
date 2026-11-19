# Plan 01: DiT export to ONNX

Status: `plan pressure-tested` (written 2026-11-18; review findings folded in, see the end of this file). Depends on: plan 00 (done). Blocks: plans 04 and 05.

Task list: [webgpu-port-tasks.md](webgpu-port-tasks.md). Runtime decision and spike numbers: [../research/webgpu-runtime-spike.md](../research/webgpu-runtime-spike.md). Golden data and tolerances: [webgpu-00-scope-and-parity.md](webgpu-00-scope-and-parity.md).

## What this plan delivers

- ONNX graphs of the `F5TTS_v1_Base` DiT that run on ORT Web's WebGPU EP:
  - A text graph, run once per chunk.
  - A step graph, run once per ODE step.
- An export script, `tools/webgpu/export_dit.py`, that builds both graphs from the upstream modules without changing them, and also exports the tiny model for CI.
- Parity against the golden `dit` and `dit_internal` stages:
  - In onnxruntime on the CPU (Python).
  - In Chrome on the RTX 5050 and the Intel GPU (TypeScript).
- Measured choices: attention form, dynamic versus fixed sequence length, and how the 1.24 GiB of fp32 step-graph weights load in a browser.

Out of scope:
- The ODE loop across steps, noise and chunking (plan 04).
- fp16, quantization and weight hosting (plan 05).
- Batch sizes above 1. Upstream single-chunk inference passes `mask=None` (`cfm.py:170`), so batch 1 needs no mask.

## What the DiT does per call (from `model/backbones/dit.py`)

- Inputs from `CFM.sample`:
  - `x`, `cond` (`step_cond`) and the text ids, which are constant within a chunk.
  - `time`, a 0-d tensor. The golden data stores it as shape (1,).
  - `cfg_infer=True`, `cache=True`.
- `time_embed(time)`: sinusoidal embedding of 256 features, then Linear, SiLU, Linear to (1, 1024). It is repeated to batch 2.
- `get_input_embed` runs twice:
  - Cond: the real `cond` and the text embedding.
  - Uncond: `cond` zeroed (`drop_audio_cond`) and the text embedding with `drop_text`.
  - The text embeddings are computed at the first step and cached for the chunk (`text_cond`, `text_uncond`).
  - With `drop_text`, the ids are zeroed after the filler mask is taken from the real ids (`dit.py:118-122`). So the uncond embedding is "token 0 everywhere, with the real filler positions masked". It is not the embedding of an all-zero id input, which +1 would turn into token 1.
- `InputEmbedding`: Linear over `cat(x, cond, text_embed)` (100 + 100 + 512 to 1024), plus `ConvPositionEmbedding`. That is two grouped Conv1d layers (kernel 31, groups 16) with Mish, applied as a residual.
- The two halves are concatenated to batch 2.
- 22 `DiTBlock`s: AdaLN with 6 modulation vectors, attention with rotary embedding on all 16 heads of 64, a gated residual, and a tanh-GELU feed-forward (1024 to 2048 to 1024).
- `norm_out` (AdaLN final), then `proj_out` (1024 to 100). The output is (2, n, 100): the cond half, then the uncond half.
- `CFM.sample` then forms `v = cond + (cond - uncond) * cfg_strength` (`cfm.py:203`). Euler takes `x_{k+1} = x_k + (t_{k+1} - t_k) * v`.
  - If `cfg_strength < 1e-5`, upstream instead makes one batch-1 call without CFG (`cfm.py:179`).
  - After the last step, `torch.where(cond_mask, cond, out)` restores the reference frames (`cfm.py:235`).
- Text embedding (`TextEmbedding`):
  - Ids plus 1, so 0 becomes filler.
  - Cut or padded to n.
  - Embedding to 512.
  - Plus a fixed sinusoidal table (`freqs_cis`, precomputed to 8192 positions).
  - 4 `ConvNeXtV2Block`s, with filler positions zeroed before and after each.
- The golden `text_ids` are (1, nt) int32 with no -1 values. They do contain real 0 ids (spaces): 37 of them in `en-multi` chunk 0.

## Decisions

### D1. Graph split (proposal: two graphs)

- **Text graph.**
  - Input: `text` int32, shape (1, n). JS supplies it already cut or padded to n with -1, as `TextEmbedding` would.
  - Outputs: `text_cond` and `text_uncond`, each (1, n, 512). Both are computed from the same input ids, with the uncond path reproducing `drop_text` as described above.
  - Run once per chunk.
- **Step graph.**
  - Inputs: `x` (1, n, 100), `step_cond` (1, n, 100), `text_cond`, `text_uncond`, `time` (1), `cfg_strength` (1) and `dt` (1).
  - Outputs, all kept on the GPU:
    - `pred` (2, n, 100): the cond and uncond halves, compared with the golden `out_cond` and `out_uncond`.
    - `v` (1, n, 100) = `cond + (cond - uncond) * cfg_strength`.
    - `x_next` (1, n, 100) = `x + dt * v`. Plan 04 can then chain steps with no separate update kernel. It is checked against the golden `s{k+1}.x`.
- Why two graphs: upstream computes the text embedding once per chunk, and a single graph would recompute it every step. The text graph is small: 5.5M parameters, 1.3M of them the embedding table.
- Alternative: one graph with an `if` on the step index. Not proposed: control flow on WebGPU is a risk, and caching is easier to reason about outside the graph.
- `cfg_strength` = 0 runs the batch-2 graph, which gives upstream's result at twice upstream's cost. A batch-1 no-CFG graph is left to plan 07 if anyone needs cfg 0.

### D2. Sequence length (proposal: dynamic n, measured against static)

- Golden chunks run from 515 to 2003 frames, and real ones go from about 300 to about 2048.
- Fixed buckets with padding would change the output, because inference runs without an attention mask: padded frames would take part in attention.
  - Buckets need the mask path, which `F5TTS_v1_Base.yaml` turns off (`attn_mask_enabled: False`). It would have to be enabled at export.
  - With it on, the key mask, the conv masking and the text `valid_pos_mask` should together reproduce the unpadded result on the valid frames. That is unverified, and the parity check would decide it.
- Proposal: export with a dynamic `n`, declared as `Dim("n", min=16, max=8192)`. The max is bounded by the 8192-position `freqs_cis` table.
- Compare with static exports at n = 1024 and 2048 on both GPUs. The comparison covers:
  - Steady-state step time.
  - The first-call latency at a new n after warming up at another n. Every chunk has a new n, so shader and pipeline creation counts.
  - ORT's `enableGraphCapture`, which only static shapes allow. It is measured, but kept separate from the 10 percent rule.
- If dynamic is less than 10 percent slower in steady state, and its new-n latency is under 1 s, keep it. Otherwise record the gap and let plan 07 decide (for example, a few static sizes plus the mask path).

### D3. Attention form (proposal: measure three, pick the fastest that passes)

The spike's export was unfused (MatMul, Softmax, MatMul). It writes out a (2, 16, n, n) score matrix, and it dominated the block at n = 2048: 3.3 times the n = 1024 time on NVIDIA, 6 to 9 times on Intel. Candidates:
1. Unfused, as dynamo exports it now (baseline).
2. `com.microsoft.MultiHeadAttention`, emitted directly by the exporter through `torch.onnx.export(..., custom_translation_table={aten.scaled_dot_product_attention: <onnxscript function>})`. The PyTorch module is not changed, and no pattern-matching graph rewrite is needed.
   - MHA takes Q, K and V as (B, S, H*D), while SDPA sees (B, H, S, D) after rotary, so the function transposes.
   - Rotary stays outside MHA.
3. ONNX `Attention` (opset 23), if the exporter emits it for `scaled_dot_product_attention` at opset 23 and ORT Web's WebGPU EP runs it.

- Each candidate must pass `dit.out` in onnxruntime CPU (step 1) before it is timed. Then it must pass in the browser (step 4) and place no float compute on the CPU.
- Timing uses the spike's method: a chained run on GPU buffers, median of 5. It covers n = 1024 and 2048 on both GPUs.

### D4. Rotary embedding (proposal: a constant table in the step graph, sliced to n)

- `RotaryEmbedding.forward_from_seq_len(n)` gives freqs (1, n, 64) from `arange(n)` and fixed inverse frequencies.
- Computing it in the graph needs an int64 `Range(n)`, then a cast. With a dynamic n, that chain is likely to land on the CPU (`WebGPU EP enable int64: 0` in the spike's log) and to add a (1, n, 64) copy.
- Proposal: precompute the table for 8192 positions as a constant (8192 x 64 fp32, 2 MB), and slice it to n, the same pattern as `freqs_cis`. If the dynamic Slice still drags a copy in, fall back to the spike's approach: freqs as a graph input built in JS.
- Revised in step 3 (2026-11-19): the constant holds cos and sin of the angles (8192 x 64 x 2, computed in float64), not the angles. The angles reach about 1,900 radians at n = 1913, and the NVIDIA GPU's shader cos/sin lose precision there; see `docs/debugging.md`. The exporter swaps upstream's `_apply_rope` for a table lookup during export only.

### D5. Weights (proposal: fp32 external data; shard before anything else)

- The step graph holds 331.6M parameters, or 1.24 GiB in fp32, counted from the checkpoint. The whole DiT is 337.1M.
- `maxBufferSize` (2 GiB on both laptop GPUs) is a per-buffer limit, and every initializer gets its own buffer, so the total is not what it constrains. The limits that matter:
  - The wasm heap. The peak is roughly the fetched ArrayBuffer plus the copy into wasm memory, within wasm32's 4 GB.
  - On the Intel GPU, shared system memory.
- Step 2 measures whether the fp32 step graph loads. Fallbacks, in order:
  1. Shard the external data into several files. ORT Web's `externalData` option takes a list.
  2. Split the step graph into two graphs of 11 blocks each.
  3. fp16 weights early, as a plan 05 task pulled forward. This is last because fp16 accuracy on the WebGPU EP is poor (spike).

### D6. Integer inputs (proposal: int32 at the graph boundary)

- The text ids are int32, cast inside the graph only where an op needs int64 (Gather accepts int32 indices).
- Shape arithmetic on dynamic n stays int64 inside ORT; see the placement criterion below.

### Placement criterion (replaces plan 00's static-shape version for these graphs)

- No float compute node is placed on the CPU, and `MemcpyTransformer` inserts no copy of an activation tensor.
- Shape-only nodes (Shape, Gather, Concat, Range and Slice on shape tensors) may sit on the CPU. Dynamic shapes keep them in the graph, where the spike's static shapes folded them away.

## Steps

### Step 1. Export script and CPU parity

- `tools/webgpu/export_dit.py`:
  - Wrapper modules `TextGraph` and `StepGraph` that call the upstream `TextEmbedding` pieces, `TimestepEmbedding`, `InputEmbedding`, `DiTBlock`s, `norm_out` and `proj_out` in the same order as `DiT.forward`.
  - Exported with `torch.onnx.export(dynamo=True)`, using a dynamic `n` (D2), external data and opset 20 (the spike's default). Opset 23 only for D3 candidate 3.
  - Loads the real checkpoint the way `make_golden.py` does, and the tiny weights from `testdata/webgpu/tiny/weights/`.
- The text graph and dynamic n: `TextEmbedding.forward` does `int(seq_len)` (`dit.py:108`), which fixes a symbolic size at export. The tensor path instead uses a data-dependent `.max().item()`.
  - So `TextGraph` calls `text_embed.text_embed`, the `freqs_cis` slice and `text_blocks` directly, with the same masking. The upstream module stays unchanged.
  - The export asserts that the graph's `n` input dimension is symbolic, not a fixed number.
- A `--debug-outputs` variant also outputs `time_embed` (1, 1024), `input_embed` and `input_embed_uncond` (1, n, 1024), every block (2, n, 1024) and `norm_out`. A failing step can then be bisected against `dit_internal`.
  - Coverage: the golden internals exist for chunk 0 only, at steps 0 and nfe/2 (steps 0 and 16; 0 and 8 for `en-epss16`; 0 and 4 for `tiny`). `text_embed` is checked only for chunk 0.
  - All other chunks and steps are checked through `dit.out` and `x_next`.
- Python check, `tools/webgpu/check_dit_ort.py` (in `.venv313`, onnxruntime CPU):
  - Feeds each golden case's `c*.dit.step_cond`, text ids (padded to n), `c*.dit.s*.x` and `t`, with `dt = t_{k+1} - t_k` taken from `c*.plan.t`.
  - Compares through `parity.py`:
    - `pred` with `out_cond` and `out_uncond` (`dit.out`).
    - `x_next` with `s{k+1}.x` (new check `dit.x_next`).
    - `v` with `out_cond + (out_cond - out_uncond) * 2.0`, computed from the golden data (new check `dit.v`).
    - The text graph with `dit_internal.text_embed`.
  - Every step of every chunk of every case.
- New checks for `tolerances.json`, set by plan 00's 10x rule from a CPU-against-CUDA measurement (the same `measure_floor.py` method):
  - `dit.v`: the CFG formula multiplies the error by up to 3.
  - `dit.x_next`: the last step of each chunk has no golden `s{k+1}.x`. It is compared with the golden `mel_out` on the generated frames (from `n_ref`), which equals `trajectory[-1]` there. No new golden tensor is needed; `parity.x_next_golden` does the lookup.
  - Neither check has a golden tensor of its own, so `checksFor` is unchanged. `parity.judge` applies their tolerances to metrics computed from golden tensors.
- Done when: both graphs, and each D3 candidate's step graph, pass `dit.out`, `dit.v`, `dit.x_next` and `dit_internal.*` for all 7 cases in onnxruntime CPU.
- Result, 2026-11-19:
  - Floors (CPU against CUDA): `dit.v` rel L2 8.4e-6 and max abs 0.0050 (tolerances 1e-4 and 0.1); `dit.x_next` 8.4e-7 and 0.00025 (1e-5 and 0.01). Existing tolerances unchanged.
  - Unfused, `mha` and `onnx23`: each passes all 1,584 comparisons across the 7 cases, internals included. Step 1 is done.
  - Weights: step graph 1,266.8 MB, text graph 37.1 MB of external data.
  - A full 7-case run takes about 20 minutes at 8 threads. The check loads one step graph at a time and disables ORT weight prepacking; with both graphs loaded it was stopped for low memory.

### Step 2. Loading the real weights in a browser (D5)

- Load the fp32 step graph with external data in Chrome on both GPUs, with a minimal page that creates the session, runs one step on golden inputs, and reads `pred` back. Record:
  - Load time from a local file server, and from the browser cache.
  - Peak JS heap and GPU memory, where Chrome's task manager or `performance.measureUserAgentSpecificMemory` shows them.
  - Whether the text graph's session can be created alongside.
- If loading fails, follow D5's fallback order and record why.
- Done when: the step graph loads and runs one step on both GPUs, or a fallback is decided and done.
- Result, 2026-11-19 (`web/bench/dit-load/`, results in `results/*.json`; Chrome, ORT Web 1.31.1, `en-short` chunk 0, n = 1913):
  - The fp32 graphs load unsharded on both GPUs, with the text graph alongside. Step 0 passes `dit.out` and `text_embed` on both. No D5 fallback is needed.
  - Memory is set by attention, not by the weights. Loading peaks at 2 to 3.5 GB of browser RAM and puts little on the GPU; ORT uploads the weights at the first run. Peaks while running one step:

    | GPU | Attention | GPU memory | Browser RAM | Step time |
    |---|---|---|---|---|
    | NVIDIA | unfused | 6.5 to 7.3 GB dedicated (of 8) | 4 to 10.6 GB | 1.1 to 1.5 s |
    | NVIDIA | `mha` | 1.7 GB dedicated | 2.5 GB | 0.9 s |
    | NVIDIA | `onnx23` | 1.6 GB dedicated | 3 GB | 25 s (likely on the CPU) |
    | Intel | unfused | 4.3 GB shared | 4.4 GB | 9 s |
    | Intel | `mha` | 1.7 GB shared | 2.3 to 3.5 GB | 2.5 to 3.7 s |

  - The unfused graph's (2, 16, n, n) scores cost about 0.47 GB per block at this n, and ORT does not keep their buffers small. This is a strong argument for `mha` in D3; step 4 still has to confirm placement and timing.
  - Load time (step graph create plus fetch): 3 to 10 s from the local server. Chrome's HTTP cache does not keep the 1.33 GB file (the second fetch transferred all of it), so the browser cache in the sense of this step means Cache Storage:
    - Filling it took 25 to 51 s (the `cache.put` of a 1.33 GB response).
    - Loading from it took 2.8 to 3.8 s, of which about 1.5 s is reading the response into an ArrayBuffer.
  - Handed to plan 05: HTTP caching does not work at this size, and Cache Storage fills slowly.
  - Times are from single runs, and the step time includes reading `pred` back. Step 4 does the real timing.
  - Memory is sampled from outside Chrome about once a second (`sample-memory.ps1`), so short peaks may be missed.

### Step 3. Browser parity runner for the DiT

- `web/src/dit/`: loads both graphs, runs the text graph once per chunk and the step graph per step, with GPU-buffer outputs.
- Tests (in `web/test/`):
  - The tiny case's exported graphs are committed (a few MB) and run in Playwright, including on SwiftShader in CI. They are compared with the tiny golden `dit` stage through the TypeScript parity runner.
  - The real cases run in a local-only Playwright script, `web/test/browser-local/` (skipped where the ONNX files and golden data are absent). It checks every step of every chunk against `dit.out`, `dit.v` and `dit.x_next` on both GPUs.
- Placement: the criterion above, read from the ORT log.
- Done when: the tiny test passes in CI, and the real cases pass on both laptop GPUs with the unfused baseline graph.
- Result, 2026-11-19:
  - Code:
    - `web/src/dit/` (`DiT`: `create`, `embedText`, `step`, `release`).
    - `web/test/dit/run-parity.ts`, the teacher-forced browser twin of `check_dit_ort.py`. It also parses node placement from ORT's verbose log and reports each check's error as a fraction of its tolerance.
    - `test/browser/dit-tiny.spec.ts` (CI).
    - `test/browser-local/dit.spec.ts` (`npm run test:browser-local`, with `PW_GPU`, `DIT_ATTENTION`, `DIT_CASES`, `DIT_DEBUG` and `DIT_MAX_STEPS`).
    - `web/vite.config.ts` serves `testdata/webgpu/` and `build/webgpu/dit/` to the dev server.
    - The TypeScript runner gained `judge` and `xNextGolden`, twins of the Python ones.
  - The first run failed narrowly on the NVIDIA GPU only, at the first steps of each chunk. The cause was in-graph rotary cos/sin; D4 is revised and `docs/debugging.md` has the bisection. The exports now carry precomputed cos/sin tables.
  - After the fix, unfused, every step of every chunk:
    - NVIDIA: all 1,228 comparisons pass across the 6 real cases. Worst `dit.v` is at 9 percent of its tolerance, worst `dit.out` at 5 percent.
    - Intel: all 1,228 pass; worst at 7.5 percent.
    - ORT CPU (`check_dit_ort.py --debug`, re-exported graphs): all 1,586 pass. The tiny graphs pass in all three attention forms.
  - Placement (tiny step graph): 10 nodes on the CPU, all int64 `Slice` and `Concat` on shapes (checked in the graph), `MemcpyTransformer` modified 0. That meets the criterion. The test fails on any other op type on the CPU.
  - Step time with the unfused graph, including reading back pred, v and x_next every step, is about 1.3 s on NVIDIA and 4 to 5 s on Intel at n near 2000. Step 4 does the real timing.
  - Not rerun after the fix: `mha` and `onnx23` on the real cases, in the browser and on the CPU. Step 4 runs every candidate through this runner anyway.

### Step 4. Attention and sequence-length variants (D2, D3)

- Every D3 candidate that passed step 1 goes through step 3's runner on both GPUs. Then each one is timed.
- `web/bench/dit-step/`: a reusable benchmark page that times the full step graph rather than one block. Plan 07 extends it; the spike stays frozen.
  - Same method as the spike: 3 warm-ups, then 20 runs and 5 chains of 32 steps, on GPU buffers.
  - Plus D2's new-n latency and the graph-capture runs.
  - Chrome on both GPUs, and Edge on NVIDIA.
- Record the results in `docs/research/webgpu-dit-export.md` (add it to `docs/index.md`).
- Done when: D2 and D3 are decided with numbers, and the chosen graphs pass step 3's runner.

### Step 5. Close out

- `docs/research/webgpu-dit-export.md` holds the variant numbers and the per-step time on both GPUs (the full step, not the spike's block projection).
- Hand off to plan 04:
  - The graph inputs and outputs, `v`, `x_next` and `dt`.
  - The `cond_mask` restore after the last step (`cfm.py:235`).
  - Golden `t` is shaped (1,), where upstream passes a 0-d tensor.
  - cfg 0 costs a full batch-2 step.
- Hand off to plan 05: sizes, load results, and whether sharding was needed.
- `docs/debugging.md` entries for export problems found. `docs/shape-ledger.md` entries for any shape or layout bugs (for example, (b, n, d) versus (b, d, n) at the conv).
- Update the task list.

## Risks

- **R1. The `custom_translation_table` route for `MultiHeadAttention`** depends on the exporter version. If it fails, a graph rewrite after export (pattern-matching the MatMul, Softmax, MatMul chain between the rotary and the transposes) is the fallback. It is more fragile.
- **R2. Dynamic shapes change ORT's kernel choice or break shape inference.** The spike used static shapes only. Also, `int()` and `.item()` calls in upstream code fix or break symbolic sizes; see step 1.
- **R3. The 1.24 GiB weight load** may hit wasm heap or browser memory limits (D5).
- **R4. Cached text embeddings must match upstream's cache semantics.** Upstream caches `text_cond` and `text_uncond` from the first DiT call of a chunk and clears them in `CFM.sample` after the chunk. The text graph is a pure function of the ids and n, so this holds as long as plan 04 reruns it whenever n or the ids change.
- **R5. `ConvPositionEmbedding` is slow** on WebGPU (about 7 percent of a step in the spike). Speeding it up is plan 07's job, but this plan's numbers should show its share.
- **R6. Node and Playwright memory.** The real-case browser tests load 1.24 GiB of weights plus golden data. They stay local and are not run in CI.
- **R7. Regenerating the golden cases for `c*.dit.x_final`.** Retired 2026-11-19: step 1 compares the last `x_next` with the golden `mel_out`, so the golden data is not regenerated.

## Pressure test, 2026-11-18

A fresh agent given only this file reviewed the first draft against the code and golden data. Findings and what changed:
- **Steps were in the wrong order.** The browser benchmarks needed both the weight-load result and a browser parity runner, which came later. The load test is now step 2 and the runner step 3, both before the variant benchmarks in step 4.
- **Dynamic n would not survive the text graph's export.** `TextEmbedding.forward` calls `int(seq_len)` (`dit.py:108`). `TextGraph` now calls the submodules directly, and the export asserts that n is symbolic.
- **"Every node on WebGPU" cannot hold with dynamic shapes**, because the int64 shape arithmetic stays in the graph. The criterion is restated: no float compute on the CPU and no activation copies. The in-graph rotary (D4) became a sliced constant table, for the same reason.
- **The debug outputs missed golden taps** (`time_embed`, `input_embed`, `input_embed_uncond`). They are added, and the internals' coverage (chunk 0, two steps) is stated.
- **`v` had no check.** `dit.v` is added, plus an `x_next` output with `dit.x_next`, so plan 04 can chain Euler steps inside the graph.
- **D2's benchmark missed new-n latency and graph capture,** and the n bound (8192, set by `freqs_cis`). All are added.
- **D5 had the `maxBufferSize` reasoning wrong and fp16 as the first fallback.** The real limits are named, and sharding is now first.
- **The `custom_translation_table` route is now D3's main route** for `MultiHeadAttention`; the graph rewrite is the fallback.
- **Smaller fixes:**
  - The text graph's uncond semantics, and the presence of real 0 ids, are spelled out.
  - Step 1's "done when" no longer depends on a later step.
  - The handoff to plan 04 gains the cfg-0 path, the `cond_mask` restore and the `t` shape.
  - D2 notes that the mask path is off in the config.
