# Task list: Python tooling, then the WebGPU port

Two bodies of work. Plan P (Python typing and tooling) comes first and is independent of the port. Plans 00 to 07 are the WebGPU port: run F5-TTS inference (voice cloning from a reference clip) in a browser on WebGPU, inference only, no training.

Model details: [../architecture.md](../architecture.md). WebGPU in the cloud container: [../research/webgpu-environment.md](../research/webgpu-environment.md).

Status values: `plan not written`, `plan drafted`, `plan pressure-tested`, `in progress`, `done`.

Every plan, once written, is saved in this folder and pressure-tested by a fresh agent (given only the plan file) before work starts. Findings get folded into the plan.

## Plan P: Python typing, formatter and linter

Plan file: [python-pyright-tooling.md](python-pyright-tooling.md). Status: `done`; the `typecheck` workflow passed in CI on 2026-11-02. Depends on: nothing. Blocks: nothing, but the port's Python export scripts (plans 01, 02, 05) should be written under it.

Where things stand (checked 2026-10-31): no type checker is configured. Ruff does lint, format and import sorting through pre-commit and the `pre-commit` GitHub workflow. Pyright 1.1.414 and mypy 2.3.1 are installed in the cloud container, but not torch. First pyright run without torch: 672 errors, 23 warnings over 47 files. Largest groups: missing imports 239, possibly-unbound variables 130, attribute access 109, invalid type arguments 69 (the ein-notation strings like `float["b n d"]`), argument types 60.

Approach decided: pyright, and port many files at once, then drive the error count to zero afterwards. Not file by file, and not gated in CI until the count is zero.

### Decisions to make first
- [x] Formatter: keep `ruff format` (decided).
- [x] Linter: keep `ruff` lint (decided). Still open inside it: whether to turn on extra rule sets (for example bugbear, pyupgrade, unused arguments) or keep the current set.
- [x] Tensor shape annotations: `jaxtyping` (decided). Tested 2026-10-31 with jaxtyping 0.3.11 on numpy arrays (torch could not be installed in the container, so torch itself is untested): pyright reads `Float[np.ndarray, "b n d"]` as a plain `ndarray`, so shapes and dtypes are not checked statically. Ruff reports F722 on shape strings and F821 on single-name dims like `"b"`; keep the `F722`/`F821` ignore, set per file in `ruff.toml`. Runtime shape checking (beartype or typeguard import hook) is optional and off by default. If missing shape checks cost enough, extend pyright itself; evidence is tracked in [../shape-ledger.md](../shape-ledger.md).
- [ ] Pyright mode to end at: `basic`, `standard` or `strict`, and for which paths. Strict only for new code (web-port tooling) is one option.
- [ ] Which paths are out of scope: `src/f5_tts/runtime/triton_trtllm/` (TensorRT-LLM APIs), `src/third_party/`, notebooks or data prep scripts.
- [ ] Policy on suppressions: only `# pyright: ignore[ruleName]` with the rule name, never bare `# type: ignore`. Comments that are temporary start with `CLAUDENOTE:` per `CLAUDE.md`.

### Setup
- [x] Make a reproducible environment with torch, torchaudio and the other runtime dependencies installed, so errors are real and not missing-import noise. Record how in `docs/debugging.md`. (Done 2026-11-01, `.venv` on Python 3.15.1 with torch 2.15.0+cpu.)
- [x] Pin the pyright version (pre-commit, dev dependency, or both). Pyright updates often add errors. (`typecheck` extra, 1.1.414, 2026-11-02.)
- [x] Add `[tool.pyright]` to `pyproject.toml`: python version 3.10, include/exclude, start in `basic` mode, report missing imports as errors only after the environment is fixed. (Done 2026-11-01.)
- [x] Record the real baseline count with torch installed, by rule. This replaces the 672 figure. (Done 2026-11-01: 457 errors, 1 warning; see the plan, step 3.)

### Bulk conversion (many files at once)
- [ ] Group files into a few batches by area: `model/` (cfm, backbones, modules, utils, dataset, trainer), `infer/` plus `api.py` and the socket files, `train/`, `eval/`, `scripts/`.
- [x] Add `jaxtyping` as a dependency (done 2026-11-01, `jaxtyping>=0.3.0`) and convert the ein-notation annotations in `model/` to it in one pass. (Done 2026-11-01.)
- [x] Add parameter and return annotations across each batch (all batches A to E done 2026-11-01) without stopping to fix every error on the way. Behaviour must not change; run the existing inference path before and after (golden output from plan 00's harness, or a fixed-seed run).
- [ ] Commit per batch so a bad batch can be reverted alone.

### Drive errors to zero
- [x] Burn down by rule category, biggest first (possibly-unbound, attribute access, argument types, then the rest), fixing real bugs separately from annotation fixes. (Done 2026-11-01.)
- [x] Any real bug found goes into `docs/debugging.md` with its fix. (Done for the 9 that pyright flagged; the unflagged ones are listed in the plan, step 5, and still open.)
- [x] Review every suppression added; each needs a rule name and a reason. (12 reviewed 2026-11-01.)
- [x] Reach zero errors in the chosen mode and paths. (0 errors, 0 warnings, basic mode, 2026-11-01.)

### Lock it in
- [x] Add pyright to pre-commit and to the GitHub workflow, failing on errors. (Done 2026-11-02 as a separate `typecheck.yaml` workflow, not pre-commit by design: pyright needs the full dependency set. Passed in CI on 2026-11-02 after two fixes.)
- [x] Add the chosen formatter and linter settings to `pyproject.toml` or `ruff.toml` and run them across the repo once, as a single formatting commit. (Settings were already in `ruff.toml`; the tree already passes `ruff format --check`, so there was nothing to reformat.)
- [x] Update `CLAUDE.md` with the commands to run (type check, lint, format) and the suppression policy. (Done 2026-11-02.)
- [x] Decide when to ratchet strictness (for example `standard` then `strict` per directory). (Decided 2026-11-02: stay on `basic`; `standard` adds 92 possibly-unbound errors; see the plan, step 6.)

# WebGPU port

Goal as above. Plans 00 to 07 below.

## Starting assumptions

These are decisions to confirm in plan 00 before anything else builds on them.

- Model scope: `F5TTS_v1_Base` only (DiT, Vocos, 24 kHz, 100 mels, hop 256). Other configs later.
- Runtime: ONNX Runtime Web with the WebGPU execution provider. Hand-written WGSL only for ops it cannot do.
- ODE loop, chunking, cross-fade and text front end live in JS/TS.
- Language scope: English and Chinese, as the pretrained model supports.
- No in-browser ASR in the first version; the user supplies the reference text.
- App code stays MIT for now. Revisit the app licence when the weights are published (decided).
- We publish the converted ONNX weights in our own Hugging Face repo (decided); the browser downloads from there. Not LFS in a GitHub repo: bandwidth quota, and Pages does not serve LFS content.

## Plans

| # | Plan (file to be written) | Depends on | Status |
|---|---|---|---|
| 00 | [Scope, runtime decision and parity harness](webgpu-00-scope-and-parity.md) | none | done 2026-11-18 (D3: ORT Web; golden data, tolerances, parity runner and CI in place) |
| 01 | [DiT export to ONNX](webgpu-01-dit-export.md) | 00 | done 2026-11-20 (fp32 text and step graphs; `mha` attention, dynamic n; parity on both GPUs; hand-offs to 04 and 05 in the plan) |
| 02 | [Mel spectrogram and vocoder](webgpu-02-mel-and-vocoder.md) | 00 | done 2026-11-21 (reference prep exact with pydub; mel, resampler and istft at the floor; Vocos graph on WebGPU; hand-offs to 04, 05 and 06 in the plan) |
| 03 | [Text front end](webgpu-03-text-frontend.md) | 00 | plan pressure-tested 2026-11-18 |
| 04 | [Sampler and generation pipeline](webgpu-04-sampler-pipeline.md) | 01, 02, 03 | plan not written |
| 05 | [Weight delivery, precision and quantization](webgpu-05-weights-precision.md) | 01, 02 | plan not written |
| 06 | [Web app and public API](webgpu-06-web-app.md) | 04, 05 | plan not written |
| 07 | [Performance and browser validation](webgpu-07-perf-validation.md) | 04, 05 | plan not written |

```
00 -> 01 -> 05 -> 06
 |    |      \-> 07
 |    \-> 04 -/
 +-> 02 -> 04
 +-> 03 -> 04
```

01, 02 and 03 can run in parallel once 00 is done.

## Tasks per plan

### 00 Scope, runtime decision and parity harness
- [x] Confirm the starting assumptions above, or change them. (Confirmed 2026-11-02, plan 00 D1: as written, plus v1 targets Chrome and Edge on desktop, tested on one discrete and one integrated GPU.)
- [x] Choose ORT Web versus custom WGSL, with a small spike: run one DiT block and a depthwise conv on WebGPU. (Done 2026-11-18: ORT Web; see [../research/webgpu-runtime-spike.md](../research/webgpu-runtime-spike.md).)
- [x] Pick the repo layout for web code (where it lives, build tool, language). (Done 2026-11-02, plan 00 D4: `web/` in this repo, TypeScript, Vite, Vitest, Playwright; Python tools in `tools/webgpu/`.)
- [x] Build the golden-data generator: fixed seed, fixed noise tensor, fixed inputs; dump text ids, reference mel, per-step DiT output, final mel, final wav from PyTorch. (Done 2026-11-18: `tools/webgpu/make_golden.py`, 7 cases.)
- [x] Define tolerances per stage (mel, DiT output, wav). (Done 2026-11-18: measured CPU vs CUDA floors, `testdata/webgpu/tolerances.json`.)
- [x] Write the test runner that compares any port stage to the golden data. (Done 2026-11-18: `web/test/parity/` and `tools/webgpu/parity.py`.)

### 01 DiT export to ONNX
- [x] Wrap `DiT.forward` for export (fixed input signature, CFG batch handled outside or inside). (Done 2026-11-19: `TextGraph` and `StepGraph` in `tools/webgpu/export_dit.py`; CFG batch inside the step graph, with `v` and `x_next` as outputs.)
- [x] Replace or precompute rotary embeddings so they export cleanly. (Done 2026-11-19: cos and sin precomputed in float64; in-graph trig was inaccurate on the NVIDIA GPU.)
- [x] Handle the text embedding cache (cond and uncond) in the exported graph. (Done: a separate text graph, run once per chunk.)
- [x] Settle dynamic versus bucketed sequence lengths; check the attention mask path. (Done 2026-11-19: dynamic n, no mask path needed; see `../research/webgpu-dit-export.md`.)
- [x] Export with external weight data; verify parity against golden data in onnxruntime (CPU). (Done 2026-11-19: all 7 cases pass.)
- [x] Verify the graph loads and matches on ORT Web WebGPU; list any unsupported or slow ops. (Done 2026-11-19: all real cases pass on both GPUs; opset 23 `Attention` is slow, graph capture fails.)
- [x] Attention: the dynamo export is unfused (MatMul, Softmax, MatMul) and dominates block time at 2048 frames, especially on the Intel GPU. Try a fused attention op that the WebGPU EP supports (plan 00 spike, `../research/webgpu-runtime-spike.md`). (Done 2026-11-19: `com.microsoft.MultiHeadAttention`, about 4 times less GPU memory and a little faster.)

### 02 Mel spectrogram and vocoder
- [x] Specify the exact Vocos mel (STFT params, filterbank, log/clamp) from `model/modules.py`. (Done 2026-11-21: `../research/webgpu-audio.md`.)
- [x] Implement mel extraction in JS or WGSL; match golden mel. (Done 2026-11-21: TypeScript, `web/src/audio/mel.ts`; the filterbank reproduces torchaudio's float32.)
- [x] Export the Vocos backbone to ONNX (start from `runtime/triton_trtllm/scripts/export_vocoder_to_onnx.py`). (Done 2026-11-21: `tools/webgpu/export_vocos.py`; every node on WebGPU.)
- [x] Implement inverse STFT outside ONNX; match golden wav. (Done 2026-11-21: `web/src/audio/istft.ts`, on the CPU, about 70 ms per long chunk.)
- [x] Reference-audio prep: mono, resample to 24 kHz, RMS normalise, clip to about 12 s. (Done 2026-11-21: exact with pydub for 16-bit PCM wav, `web/src/audio/prepare.ts`.)

### 03 Text front end
- [ ] Specify `convert_char_to_pinyin` behaviour precisely, including polyphones and non-Chinese text.
- [ ] Choose a JS pinyin and word-segmentation library; measure mismatches against Python on a test corpus.
- [ ] Port vocab loading (`infer/examples/vocab.txt`; index 0 is space and unknown).
- [ ] Port `chunk_text` (UTF-8 byte budget, ASCII and CJK sentence punctuation).
- [ ] Port reference-text punctuation fix-up. Note the double space: `preprocess_ref_audio_text` appends `". "` and `infer_batch_process` adds another space, so the token sequence has two spaces before the generated text (seen in the plan 00 golden data).
- [ ] Build a corpus test: token ids from JS equal token ids from Python.

### 04 Sampler and generation pipeline
- [ ] ODE Euler loop with CFG (`pred + (pred - null) * strength`).
- [ ] Sway sampling and EPSS timestep table.
- [ ] Duration estimate from reference text and audio, `fix_duration` split, short-chunk speed rule.
- [ ] Noise generation with seedable RNG; decide whether to match PyTorch noise or accept non-identical output.
- [ ] Conditioning mel assembly, output crop of reference frames, mel to vocoder.
- [ ] Per-chunk loop and cross-fade assembly; optional streaming of chunks.
- [ ] End-to-end test against golden wav with injected noise.
- [ ] Start from plan 01's hand-off (`webgpu-01-dit-export.md`, step 5): graph IO, the `cond_mask` restore, the frame-0 off-by-one, `time` shaped (1,), cfg 0 cost, per-step time budget.
- [ ] And from plan 02's (`webgpu-02-mel-and-vocoder.md`, step 6): the audio functions, the `rms`/`raised` rescale contract, and the mel layout transposes between `logMel`, the DiT and the vocoder.

### 05 Weight delivery, precision and quantization
- [ ] Hugging Face hosting (decided). The browser needs converted ONNX files, so we publish them in our own repo (decided). Derivative of CC-BY-NC weights: repo owner, CC-BY-NC licence tag, model card crediting the F5-TTS authors and the Emilia dataset, link to the original `SWivid/F5-TTS` repo, and a statement that use is non-commercial.
- [ ] Keep weights out of the app repo and its licence: the app downloads them from the Hugging Face repo, and the app README states the weights carry their own licence.
- [ ] Script that converts the original safetensors to the published ONNX files and uploads them, so the repo can be rebuilt.
- [ ] Download plan for roughly 0.7 to 1.3 GB: CORS check on the chosen URLs, progress UI, resume, file sharding.
- [ ] Cache in OPFS or IndexedDB; versioning and eviction.
- [ ] fp16 path with `shader-f16` detection; fp32 fallback. The WebGPU EP's fp16 MatMul is about 30 times less accurate than ORT CPU fp16 (rel L2 4e-3 at K = 1024), which looks like fp16 accumulation; measure the effect on audio, and try fp32 accumulation or mixed precision (plan 00 spike).
- [ ] Buffer and binding size limits across target devices.
- [ ] Try int8 and 4-bit quantization; judge quality with listening tests and the eval metrics already in `src/f5_tts/eval/`.
- [ ] Start from plan 01's hand-off (`webgpu-01-dit-export.md`, step 5): fp32 sizes (1.33 GB step, 39 MB text), no sharding needed, Chrome's HTTP cache does not keep the step file, Cache Storage fills in 25 to 51 s.
- [ ] And plan 02's: `vocos.onnx` is 54.3 MB fp32, and an fp16 export must keep the exact GELU.

### 06 Web app and public API
- [ ] Minimal demo page: reference upload or record, reference text, target text, play and download.
- [ ] Reusable module API (load, generate, cancel, progress events).
- [ ] Run inference in a Worker so the UI stays responsive.
- [ ] Decide on optional reference transcription (transformers.js Whisper) and its cost.
- [ ] Error handling: no WebGPU, out of memory, download failure.
- [ ] Start from plan 02's hand-off (`webgpu-02-mel-and-vocoder.md`, step 6): 16-bit PCM wav is exact, other formats are converted to int16, decode without resampling, refuse a silent reference.
- [ ] Detect an integrated GPU (`adapter.info`) and tell the user how to switch the browser to the discrete GPU. On Windows the page cannot ask for it: Chrome ignores `powerPreference` there (found in plan 00 step 2, see `../research/webgpu-laptop.md`).

### 07 Performance and browser validation
- [ ] Benchmark script: real-time factor, time to first audio, peak GPU memory.
- [ ] Profile per-step cost; find the slowest ops (attention, convs). Known from the plan 00 spike: `conv_pos_embed` (grouped conv, 140 to 560 GFLOPS) is about 7 percent of a step, and Edge was up to 2 times slower than Chrome on the same GPU.
- [ ] Optimisations: step count (EPSS 16 steps), batching the CFG pair, sequence bucketing.
- [ ] Browser and device matrix: Chrome, Edge, Safari, Firefox; desktop GPU, integrated GPU, mobile. On Windows laptops the integrated GPU is what a page gets by default, so it is the main case, not the fallback.
- [ ] Record limits and known failures in [../debugging.md](../debugging.md).
- [ ] Retry ORT graph capture with a newer ORT Web (fails in 1.31.1; worth under 10 percent per step at n = 1024). Consider a batch-1 step graph for cfg 0. Extend `web/bench/dit-step/`.
- [ ] The istft (about 70 ms per long chunk on the CPU) is the vocoder's largest cost: run it in a worker during the next chunk's DiT steps, or use a packed real FFT (plan 02, step 5).

## Open questions

- ~~Is exact numerical match with PyTorch required, or only equal audio quality?~~ Answered 2026-11-02 (plan 00 D2): numerical per stage with teacher-forced inputs, equal quality end to end.
- Exact attribution wording for the model card. Gate for plan 05: nothing is published until this is settled.
- App licence at publish time. MIT code with separately licensed CC-BY-NC weights is how upstream does it, so the code may not need to change as long as the weights are downloaded separately and not bundled in the app repo. If the app is hosted as a product and non-commercial use is intended, decide whether the app should say so. Revisit when plan 05 publishes.
- ~~Which browsers are in scope.~~ Answered 2026-11-02 (plan 00 D1): Chrome and Edge on desktop for v1; Safari and Firefox tested in plan 07 but not blocking.
- ~~Whether the Python package should gain a `webgpu/` export script entry point or this stays a separate project.~~ Answered 2026-11-02 (plan 00 D4): no entry point for now; export tools stay in `tools/webgpu/`, outside the `f5_tts` package.
